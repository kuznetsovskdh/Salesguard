import os, json, io
import pandas as pd
import plotly.express as px
import plotly
import psycopg2
from functools import wraps
from flask import (Flask, request, render_template, jsonify,
                   session, redirect, url_for, send_file)
from werkzeug.security import generate_password_hash, check_password_hash
from normalizer import (Normalizer, profile, detect_type, classify_column,
                        score_revenue, score_price, score_quantity,
                        score_date, score_customer, score_product, score_dimension)
from analytics.cleaning import clean_data
from analytics.features import add_features
from analytics.abc import abc_analysis
from analytics.seasonality import seasonality
from analytics.clustering import cluster_customers

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'salesguard-secret-2025')
@app.errorhandler(413)
def too_large(e):
    return render_template('index.html', uploads=[], logs=[],
        warning='Файл слишком большой. Максимальный размер — 50 МБ.'), 413

@app.errorhandler(500)
def server_error(e):
    return render_template('index.html', uploads=[], logs=[],
        warning=f'Внутренняя ошибка сервера: {str(e)}'), 500

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', uploads=[], logs=[],
        warning='Страница не найдена.'), 404


app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
UPLOAD_FOLDER = '/app/uploads'
ALLOWED = {'.csv', '.xlsx', '.xls', '.json', '.txt'}


# ── DB ────────────────────────────────────────────────────────────────────────
def get_db():
    return psycopg2.connect(
        host=os.environ['DB_HOST'], port=os.environ['DB_PORT'],
        dbname=os.environ['DB_NAME'], user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'])


# ── Auth decorators ───────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if not session.get('is_admin'):
            return render_template('login.html', error='Доступ только для администраторов'), 403
        return f(*args, **kwargs)
    return decorated


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_val(v):
    try:
        if pd.isna(v): return None
    except (TypeError, ValueError):
        pass
    return v

def get_col_debug(df):
    result = []
    for col in df.columns:
        s = df[col]
        f = profile(s)
        dtype = detect_type(f)
        role, conf = classify_column(s, col)
        raw_scores = {
            "DATE":        round(score_date(f, col), 3),
            "REVENUE":     round(score_revenue(f, col), 3),
            "PRICE":       round(score_price(f, col), 3),
            "QUANTITY":    round(score_quantity(f, col), 3),
            "CUSTOMER_ID": round(score_customer(f, col), 3),
            "PRODUCT":     round(score_product(f, col), 3),
            "DIMENSION":   round(score_dimension(f, col), 3),
        }
        prof_clean = {
            "numeric_ratio":  round(f["numeric_ratio"], 3),
            "datetime_ratio": round(f["datetime_ratio"], 3),
            "unique_ratio":   round(f["unique_ratio"], 3),
            "n_unique":       f["n_unique"],
            "null_ratio":     round(f["null_ratio"], 3),
            "mean":    round(f["mean"], 2) if f.get("mean") is not None else None,
            "std":     round(f["std"],  2) if f.get("std")  is not None else None,
            "max_val": f.get("max_val"),
            "min_val": f.get("min_val"),
            "monotonic": f.get("monotonic"),
        }
        result.append({"column": col, "role": role, "confidence": conf,
                       "dtype": dtype, "profile": prof_clean, "scores": raw_scores})
    return result

def save_debug_profiles(cur, upload_id, debug_cols):
    for d in debug_cols:
        cur.execute('''INSERT INTO debug_profiles
            (upload_id,column_name,mapped_role,confidence,dtype,profile_json,scores_json)
            VALUES (%s,%s,%s,%s,%s,%s,%s)''',
            (upload_id, d["column"], d["role"],
             round((d["confidence"] or 0)*100, 2), d["dtype"],
             json.dumps(d["profile"]), json.dumps(d["scores"])))

def build_text_log(upload, mapping, debug_cols, verdicts, stats=None):
    lines = [f"=== Upload #{upload[0]} | {upload[1]} ===",
             f"Статус: {upload[6]} | Формат: {upload[2]} | Строк: {upload[4]} | Чистых: {upload[5]}"]
    if upload[7]: lines.append(f"Ошибка: {upload[7]}")
    lines.append("")
    if stats:
        lines += [f"--- Статистика ---",
                  f"Строк: {stats['rows']} | Выручка: {stats['revenue_total']:.0f} | "
                  f"Средний чек: {stats['avg_check']:.0f} | Товаров: {stats['products']}", ""]
    lines.append("--- Маппинг колонок ---")
    for m in mapping:
        lines.append(f"  {m['col']:30s} -> {str(m['role']):15s} conf:{m['conf']}%  dtype:{m.get('dtype','?')}")
    lines.append("\n--- Debug по колонкам ---")
    for d in debug_cols:
        lines.append(f"\n[{d['column']}]  role={d['role']}  conf={d['confidence']*100:.0f}%  dtype={d['dtype']}")
        p = d['profile']
        lines.append(f"  numeric={p['numeric_ratio']} datetime={p['datetime_ratio']} "
                     f"unique={p['unique_ratio']} n_unique={p['n_unique']} null={p['null_ratio']}")
        if p.get('mean') is not None:
            lines.append(f"  mean={p['mean']} std={p['std']} max={p['max_val']} min={p['min_val']}")
        scores_sorted = sorted(d['scores'].items(), key=lambda x: x[1], reverse=True)
        lines.append("  scores: " + " | ".join(f"{r}={s}" for r,s in scores_sorted[:4]))
    if verdicts:
        lines.append(f"\n--- Мусорные значения ({len(verdicts)}) ---")
        for v in verdicts[:50]:
            lines.append(f"  col={v['col']} row={v['row']} reason={v['reason']}")
    return "\n".join(lines)

def build_logs(cur, limit=10):
    cur.execute('''SELECT id,filename,format,rows_total,rows_clean,status,error_msg,created_at
                   FROM uploads ORDER BY created_at DESC LIMIT %s''', (limit,))
    uploads = cur.fetchall()
    logs = []
    for u in uploads:
        uid = u[0]
        cur.execute('SELECT source_col,mapped_role,confidence FROM column_mapping WHERE upload_id=%s ORDER BY id', (uid,))
        mapping = [{'col':r[0],'role':r[1],'conf':int(r[2] or 0)} for r in cur.fetchall()]
        cur.execute('SELECT column_name,row_index,reason FROM value_verdicts WHERE upload_id=%s LIMIT 100', (uid,))
        verdicts = [{'col':r[0],'row':r[1],'reason':r[2]} for r in cur.fetchall()]
        rt, rc = u[3] or 0, u[4] or 0
        dropped = rt - rc
        drop_pct = round(dropped/rt*100,1) if rt else 0
        status_class = 'success' if u[5]=='success' else ('warn' if u[5]=='no_revenue' else 'error')
        logs.append({'id':uid,'filename':u[1],'format':u[2],'rows_total':rt,'rows_clean':rc,
                     'status':u[5],'error_msg':u[6],
                     'created_at':u[7].strftime('%d.%m.%Y %H:%M') if u[7] else '',
                     'status_class':status_class,'dropped':dropped,'drop_pct':drop_pct,
                     'mapping':mapping,'verdicts':verdicts})
    return logs


# ── AUTH ROUTES ───────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        conn = get_db(); cur = conn.cursor()
        cur.execute('SELECT id,password_hash,role FROM users WHERE username=%s', (username,))
        row = cur.fetchone(); conn.close()
        if row and check_password_hash(row[1], password):
            session['user_id'] = row[0]
            session['username'] = username
            session['is_admin'] = row[2] == 'admin'
            return redirect(url_for('index'))
        return render_template('login.html', error='Неверное имя пользователя или пароль')
    return render_template('login.html', error=None)

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','')
        password2 = request.form.get('password2','')
        if not username or len(username) < 3:
            return render_template('register.html', error='Имя пользователя минимум 3 символа', info=None)
        if len(password) < 4:
            return render_template('register.html', error='Пароль минимум 4 символа', info=None)
        if password != password2:
            return render_template('register.html', error='Пароли не совпадают', info=None)
        conn = get_db(); cur = conn.cursor()
        # Первый пользователь — admin
        cur.execute('SELECT COUNT(*) FROM users')
        count = cur.fetchone()[0]
        role = 'admin' if count == 0 else 'user'
        try:
            cur.execute('INSERT INTO users (username,password_hash,role) VALUES (%s,%s,%s)',
                        (username, generate_password_hash(password), role))
            conn.commit()
            conn.close()
            info = f'Аккаунт создан! {"Вы первый — вам выдана роль admin." if role=="admin" else ""}'
            return render_template('register.html', error=None, info=info)
        except Exception:
            conn.rollback(); conn.close()
            return render_template('register.html', error='Имя пользователя уже занято', info=None)
    return render_template('register.html', error=None, info=None)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ── ADMIN ROUTES ──────────────────────────────────────────────────────────────
@app.route('/admin')
@admin_required
def admin_panel():
    q = request.args.get('q','').strip()
    conn = get_db(); cur = conn.cursor()
    if q:
        cur.execute('''SELECT u.id,u.username,u.role,u.created_at,COUNT(up.id)
                       FROM users u LEFT JOIN uploads up ON up.id IS NOT NULL
                       WHERE u.username ILIKE %s GROUP BY u.id ORDER BY u.id''', (f'%{q}%',))
    else:
        cur.execute('''SELECT u.id,u.username,u.role,u.created_at,
                       (SELECT COUNT(*) FROM uploads) as uploads
                       FROM users u ORDER BY u.id''')
    users = [{'id':r[0],'username':r[1],'role':r[2],
              'created_at':r[3].strftime('%d.%m.%Y') if r[3] else '',
              'uploads':r[4]} for r in cur.fetchall()]
    cur.execute('SELECT COUNT(*) FROM uploads')
    total_uploads = cur.fetchone()[0]
    conn.close()
    msg = request.args.get('msg')
    err = request.args.get('err')
    return render_template('admin.html', users=users, q=q,
                           total_uploads=total_uploads, msg=msg, err=err)

@app.route('/admin/role', methods=['POST'])
@admin_required
def admin_change_role():
    user_id = request.form.get('user_id')
    role = request.form.get('role')
    if role not in ('admin','user'):
        return redirect(url_for('admin_panel', err='Недопустимая роль'))
    if str(user_id) == str(session.get('user_id')):
        return redirect(url_for('admin_panel', err='Нельзя изменить собственную роль'))
    conn = get_db(); cur = conn.cursor()
    cur.execute('UPDATE users SET role=%s WHERE id=%s', (role, user_id))
    conn.commit(); conn.close()
    action = 'повышен до admin' if role=='admin' else 'понижен до user'
    return redirect(url_for('admin_panel', msg=f'Пользователь #{user_id} {action}'))

@app.route('/admin/delete', methods=['POST'])
@admin_required
def admin_delete_user():
    user_id = request.form.get('user_id')
    if str(user_id) == str(session.get('user_id')):
        return redirect(url_for('admin_panel', err='Нельзя удалить себя'))
    conn = get_db(); cur = conn.cursor()
    cur.execute('DELETE FROM users WHERE id=%s', (user_id,))
    conn.commit(); conn.close()
    return redirect(url_for('admin_panel', msg=f'Пользователь #{user_id} удалён'))


# ── DOWNLOAD XLSX ─────────────────────────────────────────────────────────────
@app.route('/download/<int:upload_id>')
@login_required
def download_xlsx(upload_id):
    conn = get_db(); cur = conn.cursor()
    cur.execute('SELECT filename FROM uploads WHERE id=%s', (upload_id,))
    urow = cur.fetchone(); conn.close()
    fname = urow[0].rsplit('.',1)[0] if urow else f'upload_{upload_id}'
    # Сначала ищем сохранённый оригинальный очищенный файл
    clean_path = os.path.join(UPLOAD_FOLDER, f"cleaned_{upload_id}.xlsx")
    if os.path.exists(clean_path):
        return send_file(clean_path, as_attachment=True,
                         download_name=f'{fname}_cleaned.xlsx',
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    # Fallback: 6 стандартных колонок из sales
    conn = get_db(); cur = conn.cursor()
    cur.execute('''SELECT sale_date,product,quantity,unit_price,revenue,customer_id
                   FROM sales WHERE upload_id=%s ORDER BY id''', (upload_id,))
    rows = cur.fetchall(); conn.close()
    if not rows:
        return render_template('result.html', charts={},
            stats={'rows':0,'revenue_total':0,'avg_check':0,'products':0},
            role_map={}, upload_id=None,
            warning='Нет данных для скачивания. Возможно, файл не содержит колонки revenue.')
    df = pd.DataFrame(rows, columns=['sale_date','product','quantity','unit_price','revenue','customer_id'])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Cleaned Data')
    output.seek(0)
    return send_file(output, as_attachment=True,
                     download_name=f'{fname}_cleaned.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


# ── MAIN ROUTES ───────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def index():
    conn = get_db(); cur = conn.cursor()
    cur.execute('SELECT * FROM uploads ORDER BY created_at DESC LIMIT 20')
    uploads = cur.fetchall()
    logs = build_logs(cur, limit=10)
    conn.close()
    warning = request.args.get('warning')
    return render_template('index.html', uploads=uploads, logs=logs, warning=warning)

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files.get('file')
    if not file: return jsonify({'error': 'Файл не выбран'}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED: return jsonify({'error': f'Формат {ext} не поддерживается'}), 400
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    norm = Normalizer(); conn = get_db(); cur = conn.cursor()
    try:
        df_raw = norm.load(path)
        role_map = norm.detect_roles(df_raw)
        matrix = norm.build_verdict_matrix(df_raw, role_map)
        df_clean_raw = norm.apply_verdicts(df_raw, matrix, role_map)
        df_std = norm.to_standard(df_clean_raw, role_map)
        debug_cols = get_col_debug(df_raw)
        if 'revenue' not in df_std.columns or df_std['revenue'].isna().all():
            cur.execute('''INSERT INTO uploads (filename,format,rows_total,rows_clean,status,error_msg)
                VALUES (%s,%s,%s,%s,%s,%s) RETURNING id''',
                (file.filename,ext,len(df_raw),0,'no_revenue','Колонка revenue не найдена'))
            upload_id = cur.fetchone()[0]
            for col,(role,conf) in role_map.items():
                cur.execute('INSERT INTO column_mapping (upload_id,source_col,mapped_role,confidence) VALUES (%s,%s,%s,%s)',
                            (upload_id,col,role,round(conf*100,2)))
            save_debug_profiles(cur, upload_id, debug_cols)
            conn.commit()
            return render_template('result.html', charts={},
                stats={'rows':0,'revenue_total':0,'avg_check':0,'products':0},
                role_map=role_map, upload_id=upload_id,
                warning='Revenue не найден. Проверьте маппинг колонок ниже.')
        df = clean_data(df_std); df = add_features(df)
        cur.execute('''INSERT INTO uploads (filename,format,rows_total,rows_clean,status)
            VALUES (%s,%s,%s,%s,%s) RETURNING id''',
            (file.filename,ext,len(df_raw),len(df),'success'))
        upload_id = cur.fetchone()[0]
        # Bulk insert для производительности
        rows_to_insert = [
            (upload_id, safe_val(row.get('sale_date')),
             safe_val(row.get('product')), safe_val(row.get('quantity')),
             safe_val(row.get('unit_price')), safe_val(row.get('revenue')),
             safe_val(row.get('customer_id')))
            for _, row in df.iterrows()
        ]
        cur.executemany('''INSERT INTO sales
            (upload_id,sale_date,product,quantity,unit_price,revenue,customer_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s)''', rows_to_insert)
        # Сохраняем очищенный оригинал для скачивания
        try:
            clean_path = os.path.join(UPLOAD_FOLDER, f"cleaned_{upload_id}.xlsx")
            df_clean_raw.to_excel(clean_path, index=False)
        except Exception:
            pass
        for col,(role,conf) in role_map.items():
            cur.execute('INSERT INTO column_mapping (upload_id,source_col,mapped_role,confidence) VALUES (%s,%s,%s,%s)',
                        (upload_id,col,role,round(conf*100,2)))
        for col,verdicts in matrix.items():
            for i,v in enumerate(verdicts):
                if not v.is_valid:
                    cur.execute('INSERT INTO value_verdicts (upload_id,column_name,row_index,is_valid,reason) VALUES (%s,%s,%s,%s,%s)',
                                (upload_id,col,i,v.is_valid,v.reason))
        save_debug_profiles(cur, upload_id, debug_cols)
        conn.commit()
        charts = {}
        sea = seasonality(df)
        if not sea.empty:
            fig = px.line(sea, x='month', y='revenue', color='year', title='Выручка по месяцам')
            charts['seasonality'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        abc = abc_analysis(df)
        if not abc.empty:
            fig2 = px.bar(abc, x='product', y='revenue', color='abc_group', title='ABC-анализ товаров')
            charts['abc'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        cl = cluster_customers(df)
        if not cl.empty and 'cluster' in cl.columns:
            fig3 = px.scatter(cl, x='total_spent', y='purchase_count', color='cluster', title='Кластеры клиентов')
            charts['clusters'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
        stats = {'rows':len(df),'revenue_total':float(df['revenue'].sum()),
                 'avg_check':float(df['avg_check'].mean()) if 'avg_check' in df else 0,
                 'products':int(df['product'].nunique()) if 'product' in df and df['product'].notna().any() else 0}
        return render_template('result.html', charts=charts, stats=stats,
                               role_map=role_map, upload_id=upload_id, warning=None)
    except Exception as e:
        conn.rollback()
        try:
            cur.execute('INSERT INTO uploads (filename,status,error_msg) VALUES (%s,%s,%s)',
                        (file.filename,'error',str(e)))
            conn.commit()
        except Exception:
            pass
        return render_template('result.html', charts={},
            stats={'rows':0,'revenue_total':0,'avg_check':0,'products':0},
            role_map={}, upload_id=None, warning=f'Ошибка обработки: {str(e)}')
    finally:
        conn.close()


# ── DEBUG ROUTES (только admin) ───────────────────────────────────────────────
@app.route('/debug-history')
@admin_required
def debug_history():
    conn = get_db(); cur = conn.cursor()
    cur.execute('''SELECT u.id,u.filename,u.format,u.rows_total,u.rows_clean,u.status,u.created_at,
                   COUNT(dp.id) as col_count FROM uploads u
                   LEFT JOIN debug_profiles dp ON dp.upload_id=u.id
                   GROUP BY u.id ORDER BY u.created_at DESC LIMIT 50''')
    rows = cur.fetchall(); conn.close()
    uploads = [{'id':r[0],'filename':r[1],'format':r[2],'rows_total':r[3],'rows_clean':r[4],
                'status':r[5],'created_at':r[6].strftime('%d.%m %H:%M') if r[6] else '','col_count':r[7]}
               for r in rows]
    return render_template('debug_history.html', uploads=uploads)

@app.route('/debug-history/<int:upload_id>')
@admin_required
def debug_report(upload_id):
    conn = get_db(); cur = conn.cursor()
    cur.execute('SELECT * FROM uploads WHERE id=%s', (upload_id,))
    u = cur.fetchone()
    if not u: conn.close(); return "Not found", 404
    upload = {'id':u[0],'filename':u[1],'format':u[2],'rows_total':u[4],'rows_clean':u[5],'status':u[6],'error_msg':u[7]}
    cur.execute('''SELECT cm.source_col,cm.mapped_role,cm.confidence,dp.dtype
                   FROM column_mapping cm LEFT JOIN debug_profiles dp
                     ON dp.upload_id=cm.upload_id AND dp.column_name=cm.source_col
                   WHERE cm.upload_id=%s ORDER BY cm.id''', (upload_id,))
    mapping = [{'col':r[0],'role':r[1],'conf':int(r[2] or 0),'dtype':r[3]} for r in cur.fetchall()]
    cur.execute('SELECT column_name,mapped_role,confidence,dtype,profile_json,scores_json FROM debug_profiles WHERE upload_id=%s ORDER BY id', (upload_id,))
    debug_cols = [{'column':r[0],'role':r[1],'confidence':float(r[2] or 0)/100,'dtype':r[3],
                   'profile':json.loads(r[4] or '{}'),'scores':json.loads(r[5] or '{}')} for r in cur.fetchall()]
    cur.execute('SELECT column_name,row_index,reason FROM value_verdicts WHERE upload_id=%s ORDER BY id', (upload_id,))
    verdicts = [{'col':r[0],'row':r[1],'reason':r[2]} for r in cur.fetchall()]
    cur.execute('SELECT COUNT(*),SUM(revenue),AVG(revenue),COUNT(DISTINCT product) FROM sales WHERE upload_id=%s', (upload_id,))
    s = cur.fetchone()
    stats = {'rows':s[0] or 0,'revenue_total':float(s[1] or 0),'avg_check':float(s[2] or 0),'products':int(s[3] or 0)} if s and s[0] else None
    conn.close()
    u_raw = (u[0],u[1],u[2],None,u[4],u[5],u[6],u[7])
    text_log = build_text_log(u_raw, mapping, debug_cols, verdicts, stats)
    return render_template('debug_report.html', upload=upload, mapping=mapping,
                           debug_cols=debug_cols, verdicts=verdicts, stats=stats, text_log=text_log)

@app.route('/debug-page')
@admin_required
def debug_page():
    conn = get_db(); cur = conn.cursor()
    cur.execute('SELECT id,filename,status,created_at FROM uploads ORDER BY created_at DESC LIMIT 30')
    hist = [{'id':r[0],'filename':r[1],'status':r[2],
             'created_at':r[3].strftime('%d.%m %H:%M') if r[3] else ''} for r in cur.fetchall()]
    conn.close()
    return render_template('debug.html', filename='', report=[], history=hist,
                           log=None, mapping=[], stats=None, verdicts=[], text_log='')

@app.route('/debug', methods=['POST'])
@admin_required
def debug():
    file = request.files.get('file')
    if not file: return jsonify({'error': 'Файл не выбран'}), 400
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path); ext = os.path.splitext(file.filename)[1].lower()
    norm = Normalizer(); conn = get_db(); cur = conn.cursor()
    try:
        df_raw = norm.load(path)
        role_map = norm.detect_roles(df_raw)
        matrix = norm.build_verdict_matrix(df_raw, role_map)
        df_clean_raw = norm.apply_verdicts(df_raw, matrix, role_map)
        df_std = norm.to_standard(df_clean_raw, role_map)
        debug_cols = get_col_debug(df_raw)
        has_revenue = 'revenue' in df_std.columns and not df_std['revenue'].isna().all()
        stats = None; rows_clean = 0
        if has_revenue:
            df = clean_data(df_std); df = add_features(df)
            rows_clean = len(df); status = 'success'; err_msg = None
            stats = {'rows':len(df),'revenue_total':float(df['revenue'].sum()),
                     'avg_check':float(df['avg_check'].mean()) if 'avg_check' in df else 0,
                     'products':int(df['product'].nunique()) if 'product' in df and df['product'].notna().any() else 0}
        else:
            status = 'no_revenue'; err_msg = 'Колонка revenue не найдена'
        cur.execute('''INSERT INTO uploads (filename,format,rows_total,rows_clean,status,error_msg)
            VALUES (%s,%s,%s,%s,%s,%s) RETURNING id''',
            (file.filename,ext,len(df_raw),rows_clean,status,err_msg))
        upload_id = cur.fetchone()[0]
        for col,(role,conf) in role_map.items():
            cur.execute('INSERT INTO column_mapping (upload_id,source_col,mapped_role,confidence) VALUES (%s,%s,%s,%s)',
                        (upload_id,col,role,round(conf*100,2)))
        for col,vlist in matrix.items():
            for i,v in enumerate(vlist):
                if not v.is_valid:
                    cur.execute('INSERT INTO value_verdicts (upload_id,column_name,row_index,is_valid,reason) VALUES (%s,%s,%s,%s,%s)',
                                (upload_id,col,i,v.is_valid,v.reason))
        save_debug_profiles(cur, upload_id, debug_cols)
        conn.commit()
        # Сохраняем очищенный оригинал
        if has_revenue:
            try:
                clean_path = os.path.join(UPLOAD_FOLDER, f"cleaned_{upload_id}.xlsx")
                df_clean_raw.to_excel(clean_path, index=False)
            except Exception:
                pass
    except Exception as e:
        conn.rollback(); upload_id = None; debug_cols = []; stats = None
        rows_clean = 0; status = 'error'; err_msg = str(e)
        try:
            cur.execute('INSERT INTO uploads (filename,status,error_msg) VALUES (%s,%s,%s)',
                        (file.filename,'error',str(e)))
            conn.commit()
        except: pass
    mapping = [{'col':col,'role':role,'conf':int(round(conf*100)),
                'dtype':next((d['dtype'] for d in debug_cols if d['column']==col),'?')}
               for col,(role,conf) in role_map.items()] if 'role_map' in dir() else []
    verdicts_disp = []
    if 'matrix' in dir():
        for col,vlist in matrix.items():
            for i,v in enumerate(vlist):
                if not v.is_valid: verdicts_disp.append({'col':col,'row':i,'reason':v.reason})
    rt = len(df_raw) if 'df_raw' in dir() else 0
    dropped = rt - rows_clean
    drop_pct = round(dropped/rt*100,1) if rt else 0
    log = {'filename':file.filename,'format':ext,'rows_total':rt,'rows_clean':rows_clean,
           'status':status,'error_msg':err_msg,'created_at':'только что',
           'status_class':'success' if status=='success' else ('warn' if status=='no_revenue' else 'error'),
           'dropped':dropped,'drop_pct':drop_pct}
    u_raw = (upload_id or 0,file.filename,ext,None,rt,rows_clean,status,err_msg)
    text_log = build_text_log(u_raw, mapping, debug_cols, verdicts_disp, stats)
    cur.execute('SELECT id,filename,status,created_at FROM uploads ORDER BY created_at DESC LIMIT 30')
    hist = [{'id':r[0],'filename':r[1],'status':r[2],
             'created_at':r[3].strftime('%d.%m %H:%M') if r[3] else ''} for r in cur.fetchall()]
    conn.close()
    return render_template('debug.html', filename=file.filename, report=debug_cols,
                           history=hist, log=log, mapping=mapping, stats=stats,
                           verdicts=verdicts_disp, text_log=text_log)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
