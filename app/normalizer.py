import pandas as pd
import numpy as np
import chardet
import magic
import re
import os
from dataclasses import dataclass

ALIASES = {
    "DATE":        ["date","дата","period","период","dt","invoice","order_date",
                    "sale_date","sale_dt","rr_dt","dateFrom","dateTo","dateUpdate"],
    "REVENUE":     ["revenue","выруч","выручка","выручка_тыс","валовая","прибыль",
                    "amount","total charges","totalcharges","monthly charges",
                    "monthlycharges","charges","cltv","retail_amount","ppvz_for_pay",
                    "итого","оборот","total_charges","monthly_charges","total_amount",
                    "profit","total_revenue","total_sales","total_sale","gross",
                    "net_sales","net_revenue","sales","turnover","proceeds",
                    "вайлдберриз реализовал товар","реализовано на сумму",
                    "к перечислению за товар","выкуплено","итого к начислению"],
    "PRICE":       ["price","цена","стоим","cost","unit_price","unitprice","цена_руб",
                    "retail_price","price_per_unit","retail_price_withdisc"],
    "QUANTITY":    ["qty","quantity","объем","объём","объем_кг","кол","units_sold",
                    "count","volume","units"],
    "CUSTOMER_ID": ["customerid","customer_id","клиент","buyer","rid"],
    "PRODUCT":     ["product","товар","артикул","item","sku","subject_name",
                    "stockcode","description","наименован","product_name",
                    "артикул поставщика","артикул продавца","название товара",
                    "предмет","код товара ozon"],
    "DIMENSION":   ["country","city","region","state","gender","category","segment",
                    "канал","категория","сегмент","регион","страна","город",
                    "service","method","contract","billing","label","reason",
                    "brand","warehouse","bonus","supplier","tech","churn_reason"],
}

FORBIDDEN = {
    "DATE":     ["zip","code","score","value","churn","tenure","latitude","longitude",
                 "lat","long","senior","citizen","flag","флаг","barcode","nm_id",
                 "chrt","rrd","rid","realizationreport"],
    "REVENUE":  ["year","год","month","месяц","zip","code","tenure","flag","флаг",
                 "churn","score","label","latitude","longitude","lat","long",
                 "скидка","discount","промо","promo","senior","citizen",
                 "service","method","contract","billing","reason","partner",
                 "dependent","phone","internet","streaming","security","backup",
                 "protection","support","multiple","paperless","payment","gender",
                 "barcode","nm_id","chrt","rrd","rid","realizationreport",
                 "order_id","penalty","commission","spp","vw","reward","acquiring",
                 "rebill","logistic","return_amount","delivery_amount","additional",
                 "transaction_id","transactionid","invoice_no","invoiceno",
                 "invoice_id","order_no","order_number","seq","index","row_id"],
    "QUANTITY": ["flag","флаг","промо","promo","churn","score","label","zip",
                 "year","год","month","месяц","charges","total","revenue",
                 "price","cost","cltv","senior","citizen","tenure","latitude",
                 "longitude","lat","long","service","method","contract",
                 "commission","spp","vw","reward","rebill","penalty"],
    "PRICE":    ["zip","id","year","год","month","месяц","flag","churn",
                 "score","label","latitude","longitude","lat","long","tenure",
                 "commission","rebill","logistic","penalty","spp","vw"],
    "CUSTOMER_ID": ["month","year","charges","revenue","total","price",
                    "zip","score","label","churn","lat","long","barcode",
                    "nm_id","chrt","rrd","realizationreport"],
    "PRODUCT":  ["charges","revenue","total","price","cost","cltv","amount",
                 "latitude","longitude","lat","long","zip","score","churn",
                 "barcode","nm_id","chrt","rrd"],
}

DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%m/%d/%Y",
    "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M", "%d-%m-%Y", "%Y-%m",
    "%m/%d/%y", "%d-%b-%Y", "%Y/%m/%d", "%d/%m/%y",
    "%B %d, %Y", "%b %d, %Y", "%d %B %Y",
]
VALID_YEAR_MIN = 2000
VALID_YEAR_MAX = 2035

# ── REPORT_TYPE detection ───────────────────────────────────────────────────
REPORT_TYPE_MARKERS = {
    "wb_detail":       ["тип документа", "обоснование для оплаты"],
    "wb_dynamics":      ["заказано, шт", "выкуплено"],
    "wb_stock":         ["остаток на начало периода", "остаток на конец периода"],
    "wb_summary":       ["№ отчета", "юридическое лицо", "тип отчета"],
    "ozon_detail":      ["код товара ozon", "реализовано на сумму"],
    "ozon_stock":       ["остаток доступный к продаже"],
    "ozon_summary":     ["заказано на сумму", "к перечислению"],
    "ozon_settlement":  ["баланс после операции"],
    "ozon_akt_sverki":  ["дебет (начислено продавцу)", "кредит (оплачено"],
}

# Отчёты, пригодные для товарной аналитики (ABC/RFM/margin)
PRODUCT_LEVEL_REPORTS = {"wb_detail", "wb_dynamics", "ozon_detail"}


def detect_report_type(df: pd.DataFrame) -> str:
    """Определяет тип отчёта по маркерным колонкам. Возвращает ключ из
    REPORT_TYPE_MARKERS или 'unknown', если ни один маркер не совпал."""
    cols_lower = [str(c).lower().strip() for c in df.columns]
    cols_joined = " | ".join(cols_lower)

    best_type, best_score = "unknown", 0
    for report_type, markers in REPORT_TYPE_MARKERS.items():
        if all(m in cols_joined for m in markers):
            score = len(markers)
            if score > best_score:
                best_type, best_score = report_type, score

    return best_type if best_score > 0 else "unknown"


def is_product_level_report(report_type: str) -> bool:
    return report_type in PRODUCT_LEVEL_REPORTS


@dataclass
class ValueVerdict:
    raw: object
    is_valid: bool
    detected_type: str
    confidence: float = 1.0
    reason: str = ""


# ── Profiling ──────────────────────────────────────────────────────────────────
def profile(s: pd.Series) -> dict:
    # Для производительности сэмплируем большие колонки
    if len(s) > 10000:
        s = s.sample(n=10000, random_state=42)
    s_str = s.astype(str).str.strip()
    numeric = pd.to_numeric(s, errors="coerce")
    f = {}
    f["n"] = len(s)
    f["null_ratio"] = s.isna().mean() + s_str.eq("").mean()
    f["numeric_ratio"] = numeric.notna().mean()
    f["unique_ratio"] = s.nunique(dropna=True) / max(1, f["n"])
    f["n_unique"] = s.nunique(dropna=True)
    f["avg_len"] = s_str.map(len).mean()

    # datetime только если значения НЕ числовые
    if f["numeric_ratio"] < 0.5:
        dt = pd.to_datetime(s, errors="coerce")
        dt_ratio = dt.notna().mean()
        # sanity check: год должен быть в разумном диапазоне
        if dt_ratio > 0.5:
            valid_years = dt.dropna().dt.year
            if not valid_years.empty:
                year_ok = ((valid_years >= VALID_YEAR_MIN) &
                           (valid_years <= VALID_YEAR_MAX)).mean()
                dt_ratio = dt_ratio * year_ok
        f["datetime_ratio"] = dt_ratio
    else:
        f["datetime_ratio"] = 0.0

    if f["numeric_ratio"] > 0.8:
        x = numeric.dropna().astype(float)
        f["mean"] = float(x.mean())
        f["std"] = float(x.std())
        f["q1"] = float(x.quantile(0.25))
        f["q3"] = float(x.quantile(0.75))
        f["iqr"] = f["q3"] - f["q1"]
        f["neg_ratio"] = float((x < 0).mean())
        f["zeros_ratio"] = float((x == 0).mean())
        f["monotonic"] = bool(x.is_monotonic_increasing or x.is_monotonic_decreasing)
        f["max_val"] = float(x.max())
        f["min_val"] = float(x.min())
    else:
        f["mean"] = f["std"] = f["iqr"] = f["neg_ratio"] = None
        f["zeros_ratio"] = f["max_val"] = f["min_val"] = None
        f["monotonic"] = False
    return f


# ── TYPE detection ─────────────────────────────────────────────────────────────
def detect_type(f: dict) -> str:
    if f["datetime_ratio"] > 0.7 and f["numeric_ratio"] < 0.5:
        return "datetime"

    if f["numeric_ratio"] > 0.8:
        mn = f.get("min_val", 0) or 0
        mx = f.get("max_val", 0) or 0
        mean = f.get("mean", 0) or 0

        # Unix timestamp в наносекундах или миллисекундах
        if mean > 1e12:
            return "unix_timestamp"

        # Географические координаты: требуем ОТРИЦАТЕЛЬНЫЕ значения
        # (исключает Monthly Charges, Churn Score и т.п.)
        if mn < 0 and -180 <= mn and mx <= 180:
            return "geo"

        # Бинарный флаг
        if mx <= 1 and mn >= 0 and f["n_unique"] <= 2:
            return "flag"

        # Месяц
        if mx <= 12 and mn >= 1 and f["n_unique"] <= 12:
            return "month_like"

        # Год
        if 2000 <= mn and mx <= 2035 and f["n_unique"] <= 10:
            return "year_like"

        # Score 0-100
        if mx <= 100 and mn >= 0 and f["n_unique"] <= 101 and (f["std"] or 0) < 40:
            return "score_like"

        # ID с очень высокой уникальностью
        if f["unique_ratio"] > 0.98 and not f["monotonic"]:
            return "id_or_key"

        return "numeric"

    if f["n_unique"] < 50:
        return "categorical"
    return "text"


# ── Name scoring ───────────────────────────────────────────────────────────────
def name_score(name: str, role: str) -> float:
    n = name.lower().replace(" ", "_")
    aliases = ALIASES.get(role, [])
    best = 0.0
    for a in aliases:
        a_norm = a.lower().replace(" ", "_")
        if a_norm == n:
            best = max(best, 1.0)
        elif a_norm in n or n in a_norm:
            best = max(best, 0.75)
    return best


# ── Type score (самый сильный сигнал) ─────────────────────────────────────────
def type_score(dtype: str, role: str) -> float:
    matrix = {
        "DATE":        {"datetime": 1.0, "unix_timestamp": 0.0},
        "REVENUE":     {"numeric": 1.0, "geo": 0.0, "unix_timestamp": 0.0,
                        "flag": 0.0, "year_like": 0.0, "month_like": 0.0},
        "PRICE":       {"numeric": 1.0, "geo": 0.0, "unix_timestamp": 0.0,
                        "flag": 0.0, "year_like": 0.0, "month_like": 0.0},
        "QUANTITY":    {"numeric": 1.0, "geo": 0.0, "unix_timestamp": 0.0,
                        "flag": 0.0, "year_like": 0.0},
        "CUSTOMER_ID": {"id_or_key": 1.0, "text": 0.9, "categorical": 0.7,
                        "geo": 0.0, "unix_timestamp": 0.0},
        "PRODUCT":     {"categorical": 1.0, "text": 0.9, "numeric": 0.0,
                        "geo": 0.0, "unix_timestamp": 0.0},
        "DIMENSION":   {"categorical": 1.0, "text": 0.8, "flag": 0.6,
                        "score_like": 0.6, "month_like": 0.4,
                        "geo": 0.3, "numeric": 0.3},
    }
    return matrix.get(role, {}).get(dtype, 0.5)


# ── Distribution score ─────────────────────────────────────────────────────────
def distribution_score(f: dict, role: str) -> float:
    if role == "REVENUE":
        std = f.get("std") or 0
        return 1.0 if std > 50 else (0.6 if std > 10 else 0.2)
    if role == "QUANTITY":
        mx = f.get("max_val") or 0
        return 1.0 if 0 < mx < 10000 else 0.3
    if role == "PRICE":
        mean = f.get("mean") or 0
        return 1.0 if 0 < mean < 50000 else 0.3
    if role == "CUSTOMER_ID":
        return 1.0 if f["unique_ratio"] > 0.9 else 0.2
    return 0.5


# ── Constraints ────────────────────────────────────────────────────────────────
def apply_constraints(name: str, role: str, f: dict, dtype: str) -> float:
    n = name.lower()
    for k in FORBIDDEN.get(role, []):
        # word-boundary: 'month' не должен блокировать 'monthly charges'
        if re.search(r'(?<![a-z])' + re.escape(k) + r'(?![a-z])', n):
            return 0.0
    # Структурные блоки
    if role == "DATE" and dtype != "datetime":
        return 0.0
    if role == "REVENUE" and dtype in ("geo", "unix_timestamp", "flag",
                                        "year_like", "month_like", "id_or_key"):
        return 0.0
    if role == "PRICE" and dtype in ("geo", "unix_timestamp", "flag",
                                      "year_like", "month_like"):
        return 0.0
    if role == "QUANTITY" and dtype in ("geo", "unix_timestamp", "flag",
                                         "year_like", "categorical", "text"):
        return 0.0
    if role == "PRODUCT" and dtype in ("geo", "unix_timestamp", "numeric",
                                        "score_like"):
        return 0.0
    if role == "CUSTOMER_ID" and f["unique_ratio"] < 0.5:
        return 0.0
    return 1.0


# ── Scoring functions ──────────────────────────────────────────────────────────
def score_revenue(f, name):
    return (0.40 * name_score(name, "REVENUE") +
            0.25 * (1 if (f.get("numeric_ratio") or 0) > 0.9 else 0) +
            0.20 * distribution_score(f, "REVENUE") +
            0.10 * (1 if (f.get("neg_ratio") or 1) < 0.3 else 0) +
            0.05 * (1 if f["unique_ratio"] < 0.99 else 0))

def score_price(f, name):
    return (0.45 * name_score(name, "PRICE") +
            0.30 * (1 if (f.get("numeric_ratio") or 0) > 0.9 else 0) +
            0.15 * distribution_score(f, "PRICE") +
            0.10 * (1 if (f.get("neg_ratio") or 1) < 0.1 else 0))

def score_quantity(f, name):
    return (0.45 * name_score(name, "QUANTITY") +
            0.30 * (1 if (f.get("numeric_ratio") or 0) > 0.9 else 0) +
            0.15 * distribution_score(f, "QUANTITY") +
            0.10 * (1 if f["n_unique"] < 1000 else 0))

def score_date(f, name):
    return (0.70 * (1 if f["datetime_ratio"] > 0.7 else 0) +
            0.30 * name_score(name, "DATE"))

def score_customer(f, name):
    return (0.50 * name_score(name, "CUSTOMER_ID") +
            0.30 * (1 if f["unique_ratio"] > 0.8 else 0) +
            0.10 * (1 if not f["monotonic"] else 0) +
            0.10 * (1 if f["null_ratio"] < 0.05 else 0))

def score_product(f, name):
    return (0.55 * name_score(name, "PRODUCT") +
            0.25 * (1 if (f.get("numeric_ratio") or 1) < 0.5 else 0) +
            0.20 * (1 if 2 < f["n_unique"] < 10000 else 0))

def score_dimension(f, name):
    return (0.40 * name_score(name, "DIMENSION") +
            0.30 * (1 if f["n_unique"] < 200 else 0) +
            0.20 * (1 if (f.get("numeric_ratio") or 1) < 0.3 else 0) +
            0.10 * (1 if f["unique_ratio"] < 0.7 else 0))


# ── Classify single column ─────────────────────────────────────────────────────
def classify_column(s: pd.Series, name: str):
    f = profile(s)
    dtype = detect_type(f)

    # id_or_key override: если имя явно указывает на финансовую роль — это не ID
    if dtype == "id_or_key":
        for fin_role in ("REVENUE", "PRICE", "QUANTITY"):
            if name_score(name, fin_role) >= 0.75:
                dtype = "numeric"
                break

    # Адаптивный порог для DATE: малые датасеты + говорящее имя
    if dtype != "datetime" and f["datetime_ratio"] > 0.4 and f["numeric_ratio"] < 0.5:
        if name_score(name, "DATE") >= 0.75:
            dtype = "datetime"

    raw = {
        "DATE":        score_date(f, name),
        "REVENUE":     score_revenue(f, name),
        "PRICE":       score_price(f, name),
        "QUANTITY":    score_quantity(f, name),
        "CUSTOMER_ID": score_customer(f, name),
        "PRODUCT":     score_product(f, name),
        "DIMENSION":   score_dimension(f, name),
    }

    # Взвешенное: 50% type + 30% name/stats + 20% distribution
    weighted = {}
    for role, raw_s in raw.items():
        ts = type_score(dtype, role)
        ds = distribution_score(f, role)
        constraint = apply_constraints(name, role, f, dtype)
        weighted[role] = (0.50 * ts + 0.30 * raw_s + 0.20 * ds) * constraint

    role = max(weighted, key=weighted.get)
    conf = weighted[role]

    if conf < 0.25:
        return None, 0.0
    return role, round(conf, 2)


# ── Cross-column normalization ─────────────────────────────────────────────────
def resolve_roles(df: pd.DataFrame) -> dict:
    """Классифицирует все колонки + cross-column resolution."""
    raw_map = {}
    for col in df.columns:
        role, conf = classify_column(df[col], col)
        raw_map[col] = (role, conf)

    # Если revenue не найден — берём колонку с наибольшим raw revenue score
    revenue_cols = [(c, conf) for c, (r, conf) in raw_map.items() if r == "REVENUE"]
    if not revenue_cols:
        candidates = []
        for col in df.columns:
            f = profile(df[col])
            dtype = detect_type(f)
            if dtype == "numeric" and apply_constraints(col, "REVENUE", f, dtype) == 1.0:
                s = score_revenue(f, col)
                if s > 0.3:
                    candidates.append((col, s))
        if candidates:
            best_col, best_score = max(candidates, key=lambda x: x[1])
            raw_map[best_col] = ("REVENUE", round(best_score, 2))

    return raw_map


# ── Value verdict ──────────────────────────────────────────────────────────────
def check_value(val, role: str) -> ValueVerdict:
    if pd.isna(val) or str(val).strip() == "":
        return ValueVerdict(val, False, "NULL", 0.0, "пустое значение")
    if role in ("REVENUE", "QUANTITY", "PRICE"):
        try:
            fv = float(val)
        except (ValueError, TypeError):
            return ValueVerdict(val, False, "STR", 0.0, "не числовое")
        if fv < 0:
            return ValueVerdict(val, False, "FLOAT", 0.3, "отрицательное")
        return ValueVerdict(val, True, "FLOAT", 1.0)
    if role == "DATE":
        for fmt in DATE_FORMATS:
            try:
                pd.to_datetime(val, format=fmt)
                return ValueVerdict(val, True, "DATE", 1.0)
            except Exception:
                pass
        return ValueVerdict(val, False, "STR", 0.0, "не является датой")
    return ValueVerdict(val, True, "ANY", 0.8)


def iqr_pass(verdicts: list) -> list:
    nums = [float(v.raw) for v in verdicts
            if v.is_valid and v.detected_type == "FLOAT"]
    if len(nums) < 30:
        return verdicts
    q1, q3 = np.percentile(nums, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    result = []
    for v in verdicts:
        if v.is_valid and v.detected_type == "FLOAT":
            fv = float(v.raw)
            if fv < lo or fv > hi:
                result.append(ValueVerdict(v.raw, False, "FLOAT", 0.1,
                    f"IQR выброс [{lo:.1f}, {hi:.1f}]"))
            else:
                result.append(v)
        else:
            result.append(v)
    return result


# ── Normalizer ─────────────────────────────────────────────────────────────────
class Normalizer:
    ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".txt"}
    MAX_SIZE_BYTES = 50 * 1024 * 1024

    def load(self, filepath: str) -> pd.DataFrame:
        self._check_size(filepath)
        fmt = self._detect_format(filepath)
        enc = self._detect_encoding(filepath)
        df = self._read(filepath, fmt, enc)
        df = self._fix_single_column_header(df, filepath, fmt, enc)
        df = self._clean_currency(df)
        return df

    def _check_size(self, path):
        if os.path.getsize(path) > self.MAX_SIZE_BYTES:
            raise ValueError("Файл превышает 50 МБ")

    def _detect_format(self, path: str) -> str:
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("xlsx", "xls"):
            return "xlsx"
        if ext == "json":
            return "json"
        if ext in ("csv", "txt"):
            return "csv"
        try:
            mime = magic.from_file(path, mime=True)
            if "spreadsheet" in mime or "excel" in mime:
                return "xlsx"
            if "json" in mime:
                return "json"
        except Exception:
            pass
        return "csv"

    def _detect_encoding(self, path: str) -> str:
        with open(path, "rb") as f:
            raw = f.read(32768)
        return chardet.detect(raw).get("encoding") or "utf-8"

    def _read_xlsx_smart_header(self, path: str) -> pd.DataFrame:
        """Ozon/WB отчёты часто имеют merged-заголовок или преамбулу перед
        реальной шапкой таблицы. Сдвигаем header построчно (до 10 строк),
        пока доля колонок 'Unnamed:' не упадёт ниже 30%."""
        for header_row in range(0, 10):
            try:
                df = pd.read_excel(path, header=header_row)
            except Exception:
                continue
            if len(df.columns) == 0:
                continue
            unnamed_ratio = sum(
                str(c).startswith("Unnamed:") for c in df.columns
            ) / len(df.columns)
            if unnamed_ratio < 0.3:
                return df
        # fallback: ничего не подошло — возвращаем header=0 как раньше
        return pd.read_excel(path)

    def _read(self, path, fmt, enc) -> pd.DataFrame:
        if fmt == "xlsx":
            return self._read_xlsx_smart_header(path)
        if fmt == "json":
            return pd.read_json(path)
        # Список кодировок для попыток — всегда включаем надёжные fallback
        encodings = list(dict.fromkeys([enc, "utf-8", "latin-1", "cp1251", "utf-8-sig"]))
        for sep in [",", ";", "\t", "|"]:
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, sep=sep, encoding=encoding,
                                     nrows=5, on_bad_lines="skip")
                    if len(df.columns) > 1:
                        return pd.read_csv(path, sep=sep, encoding=encoding,
                                           on_bad_lines="skip", low_memory=False)
                except UnicodeDecodeError:
                    continue
                except Exception:
                    break
        # Финальный fallback: latin-1 читает любой байт
        try:
            df = pd.read_csv(path, sep=None, engine="python",
                             encoding="latin-1", on_bad_lines="skip",
                             low_memory=False)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass
        raise ValueError("Не удалось определить разделитель CSV")

    def _clean_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Убирает символы валют из числовых колонок. Пропускает большие df."""
        if len(df) > 10000:
            return df  # Пропускаем для производительности
        currency_re = r"[₹$€£¥₩₽]"
        sample_size = min(100, len(df))
        for col in df.columns:
            if df[col].dtype != object:
                continue
            # Быстрая проверка на образце
            sample = df[col].dropna().head(sample_size).astype(str)
            if not sample.str.contains(r"[₹$€£¥₩₽]", regex=True).any():
                continue
            cleaned = df[col].astype(str).str.replace(currency_re, "", regex=True)
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.notna().mean() > 0.5:
                df[col] = cleaned
        return df

    def _fix_single_column_header(self, df: pd.DataFrame,
                                   filepath: str, fmt: str, enc: str) -> pd.DataFrame:
        """Если прочитан 1 столбец с запятыми — данные в CSV-формате внутри ячеек."""
        if len(df.columns) != 1:
            return df
        col_name = str(df.columns[0])
        if "," not in col_name:
            return df
        # Вариант 1: каждая ячейка — comma-separated строка
        try:
            col_names = [c.strip() for c in col_name.split(",")]
            if len(col_names) > 1:
                rows = []
                for val in df.iloc[:, 0]:
                    parts = [p.strip() for p in str(val).split(",")]
                    if len(parts) == len(col_names):
                        rows.append(parts)
                if rows:
                    return pd.DataFrame(rows, columns=col_names)
        except Exception:
            pass
        # Вариант 2: перечитать файл без header
        try:
            if fmt == "xlsx":
                df2 = pd.read_excel(filepath, header=None)
            else:
                df2 = pd.read_csv(filepath, header=None, encoding=enc)
            df2.columns = df2.iloc[0].astype(str).str.strip()
            df2 = df2[1:].reset_index(drop=True)
            if len(df2.columns) > 1:
                return df2
        except Exception:
            pass
        return df

    def detect_roles(self, df: pd.DataFrame) -> dict:
        return resolve_roles(df)

    def build_verdict_matrix(self, df, role_map) -> dict:
        matrix = {}
        for col, (role, _) in role_map.items():
            if role is None:
                continue
            verdicts = [check_value(v, role) for v in df[col]]
            if role in ("REVENUE", "QUANTITY", "PRICE"):
                verdicts = iqr_pass(verdicts)
            matrix[col] = verdicts
        return matrix

    def apply_verdicts(self, df, matrix, role_map) -> pd.DataFrame:
        key_cols = [c for c, (r, _) in role_map.items()
                    if r in ("REVENUE", "DATE")]
        mask = pd.Series([True] * len(df), index=df.index)
        for col in key_cols:
            if col in matrix:
                col_ok = pd.Series(
                    [v.is_valid for v in matrix[col]], index=df.index)
                mask &= col_ok
        bad_pct = (~mask).sum() / max(len(df), 1)
        if bad_pct < 0.05:
            return df[mask].reset_index(drop=True)
        df_clean = df.copy()
        for col, (role, _) in role_map.items():
            if role in ("REVENUE", "QUANTITY", "PRICE") and col in matrix:
                bad_idx = [i for i, v in enumerate(matrix[col]) if not v.is_valid]
                med = pd.to_numeric(df_clean[col], errors="coerce").median()
                df_clean.loc[bad_idx, col] = med
        return df_clean

    def to_standard(self, df, role_map) -> pd.DataFrame:
        def get(role):
            cols = [(c, conf) for c, (r, conf) in role_map.items() if r == role]
            if not cols:
                return None
            # Выбираем колонку с наивысшим confidence
            best_col = max(cols, key=lambda x: x[1])[0]
            return df[best_col]

        std = pd.DataFrame()
        std["sale_date"] = pd.to_datetime(get("DATE"), errors="coerce")
        std["product"] = get("PRODUCT")
        q = get("QUANTITY")
        pr = get("PRICE")
        rev = get("REVENUE")
        if rev is not None:
            std["revenue"] = pd.to_numeric(rev, errors="coerce")
        elif q is not None and pr is not None:
            std["revenue"] = (pd.to_numeric(q, errors="coerce") *
                              pd.to_numeric(pr, errors="coerce"))
        std["quantity"] = pd.to_numeric(q, errors="coerce") if q is not None else None
        std["unit_price"] = pd.to_numeric(pr, errors="coerce") if pr is not None else None
        std["customer_id"] = get("CUSTOMER_ID")
        return std
