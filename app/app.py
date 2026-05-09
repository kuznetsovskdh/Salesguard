import os, json
import pandas as pd
import plotly.express as px
import plotly
import psycopg2
from flask import Flask, request, render_template, jsonify
from normalizer import Normalizer, profile, detect_type, classify_column
from normalizer import score_revenue, score_price, score_quantity
from normalizer import score_date, score_customer, score_product, score_dimension
from analytics.cleaning import clean_data
from analytics.features import add_features
from analytics.abc import abc_analysis
from analytics.seasonality import seasonality
from analytics.clustering import cluster_customers

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
UPLOAD_FOLDER = '/app/uploads'
ALLOWED = {'.csv', '.xlsx', '.xls', '.json', '.txt'}

def get_db():
    return psycopg2.connect(
        host=os.environ['DB_HOST'], port=os.environ['DB_PORT'],
        dbname=os.environ['DB_NAME'], user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'])

def build_logs(cur, limit=10):
    cur.execute('''
        SELECT id, filename, format, rows_total, rows_clean,
               status, error_msg, created_at
        FROM uploads ORDER BY created_at DESC LIMIT %s''', (limit,))
    uploads = cur.fetchall()
    logs = []
    for u in uploads:
        uid, fname, fmt, rt, rc, status, err, ts = u
        # маппинг
        cur.execute('''SELECT source_col, mapped_role, confidence
                       FROM column_mapping WHERE upload_id=%s
                       ORDER BY id''', (uid,))
        mapping = [{'col': r[0], 'role': r[1], 'conf': int(r[2] or 0)}
                   for r in cur.fetchall()]
        # вердикты
        cur.execute('''SELECT column_name, row_index, reason
                       FROM value_verdicts WHERE upload_id=%s
                       ORDER BY id LIMIT 100''', (uid,))
        verdicts = [{'col': r[0], 'row': r[1], 'reason': r[2]}
                    for r in cur.fetchall()]

        dropped = (rt or 0) - (rc or 0)
        drop_pct = round(dropped / rt * 100, 1) if rt else 0
        status_class = ('success' if status == 'success'
                        else 'warn' if status == 'no_revenue'
                        else 'error')
        logs.append({
            'id': uid, 'filename': fname, 'format': fmt,
            'rows_total': rt, 'rows_clean': rc,
            'status': status, 'error_msg': err,
            'created_at': ts.strftime('%d.%m.%Y %H:%M') if ts else '',
            'status_class': status_class,
            'dropped': dropped, 'drop_pct': drop_pct,
            'mapping': mapping, 'verdicts': verdicts,
        })
    return logs

@app.route('/')
def index():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM uploads ORDER BY created_at DESC LIMIT 20')
    uploads = cur.fetchall()
    logs = build_logs(cur, limit=10)
    conn.close()
    return render_template('index.html', uploads=uploads, logs=logs)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Файл не выбран'}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify({'error': f'Формат {ext} не поддерживается'}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    norm = Normalizer()
    conn = get_db()
    cur = conn.cursor()

    try:
        df_raw = norm.load(path)
        role_map = norm.detect_roles(df_raw)
        matrix = norm.build_verdict_matrix(df_raw, role_map)
        df_clean_raw = norm.apply_verdicts(df_raw, matrix, role_map)
        df_std = norm.to_standard(df_clean_raw, role_map)

        if 'revenue' not in df_std.columns or df_std['revenue'].isna().all():
            cur.execute('''INSERT INTO uploads
                (filename, format, rows_total, rows_clean, status, error_msg)
                VALUES (%s,%s,%s,%s,%s,%s) RETURNING id''',
                (file.filename, ext, len(df_raw), 0, 'no_revenue',
                 'Колонка revenue не найдена'))
            upload_id = cur.fetchone()[0]
            for col, (role, conf) in role_map.items():
                cur.execute('''INSERT INTO column_mapping
                    (upload_id, source_col, mapped_role, confidence)
                    VALUES (%s,%s,%s,%s)''',
                    (upload_id, col, role, round(conf * 100, 2)))
            conn.commit()
            return render_template('result.html',
                charts={}, stats={'rows': 0, 'revenue_total': 0,
                'avg_check': 0, 'products': 0},
                role_map=role_map, upload_id=upload_id,
                warning='Revenue не найден. Проверьте маппинг колонок ниже.')

        df = clean_data(df_std)
        df = add_features(df)

        cur.execute('''INSERT INTO uploads
            (filename, format, rows_total, rows_clean, status)
            VALUES (%s,%s,%s,%s,%s) RETURNING id''',
            (file.filename, ext, len(df_raw), len(df), 'success'))
        upload_id = cur.fetchone()[0]

        for _, row in df.iterrows():
            cur.execute('''INSERT INTO sales
                (upload_id, sale_date, product, quantity, unit_price, revenue, customer_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s)''',
                (upload_id, row.get('sale_date'), row.get('product'),
                 row.get('quantity'), row.get('unit_price'),
                 row.get('revenue'), row.get('customer_id')))

        for col, (role, conf) in role_map.items():
            cur.execute('''INSERT INTO column_mapping
                (upload_id, source_col, mapped_role, confidence)
                VALUES (%s,%s,%s,%s)''',
                (upload_id, col, role, round(conf * 100, 2)))

        for col, verdicts in matrix.items():
            for i, v in enumerate(verdicts):
                if not v.is_valid:
                    cur.execute('''INSERT INTO value_verdicts
                        (upload_id, column_name, row_index, is_valid, reason)
                        VALUES (%s,%s,%s,%s,%s)''',
                        (upload_id, col, i, v.is_valid, v.reason))
        conn.commit()

        charts = {}
        sea = seasonality(df)
        if not sea.empty:
            fig = px.line(sea, x='month', y='revenue', color='year',
                          title='Выручка по месяцам')
            charts['seasonality'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        abc = abc_analysis(df)
        if not abc.empty:
            fig2 = px.bar(abc, x='product', y='revenue', color='abc_group',
                          title='ABC-анализ товаров')
            charts['abc'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        cl = cluster_customers(df)
        if not cl.empty:
            fig3 = px.scatter(cl, x='total_spent', y='purchase_count',
                              color='cluster', title='Кластеры клиентов')
            charts['clusters'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

        stats = {
            'rows': len(df),
            'revenue_total': float(df['revenue'].sum()),
            'avg_check': float(df['avg_check'].mean()) if 'avg_check' in df else 0,
            'products': int(df['product'].nunique()) if 'product' in df and df['product'].notna().any() else 0,
        }

        return render_template('result.html', charts=charts, stats=stats,
                               role_map=role_map, upload_id=upload_id,
                               warning=None)

    except Exception as e:
        conn.rollback()
        cur.execute('''INSERT INTO uploads (filename, status, error_msg)
            VALUES (%s,%s,%s)''', (file.filename, 'error', str(e)))
        conn.commit()
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()

@app.route('/debug-page')
def debug_page():
    return render_template('debug.html', filename='', report=[])

@app.route('/debug', methods=['POST'])
def debug():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Файл не выбран'}), 400
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    norm = Normalizer()
    df = norm.load(path)
    report = []
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
        report.append({
            "column": col, "role": role, "conf": conf, "dtype": dtype,
            "profile": {
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
            },
            "raw_scores": raw_scores,
        })
    return render_template('debug.html', filename=file.filename, report=report)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
