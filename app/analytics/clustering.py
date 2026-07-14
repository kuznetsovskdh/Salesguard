import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_customers(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    if 'customer_id' not in df.columns or df['customer_id'].isna().all():
        return pd.DataFrame()
    cust = df.groupby('customer_id').agg(
        total_spent=('revenue', 'sum'),
        purchase_count=('revenue', 'count')
    ).reset_index()
    if len(cust) < n_clusters:
        return cust
    X = StandardScaler().fit_transform(cust[['total_spent', 'purchase_count']])
    cust['cluster'] = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)
    return cust


def rfm_sku_analysis(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Товарный RFM: Recency/Frequency/Monetary по SKU (product), не по клиенту.

    Recency  - дни с последней продажи SKU
    Frequency - число транзакций продажи SKU за период
    Monetary  - суммарная выручка SKU

    Строки-возвраты (is_return=True) исключаются из расчёта, если колонка
    присутствует (иначе используются все строки).
    """
    if 'product' not in df.columns or df['product'].isna().all():
        return pd.DataFrame()
    if 'sale_date' not in df.columns or 'revenue' not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    if 'is_return' in work.columns:
        work = work[~work['is_return']]

    if work.empty:
        return pd.DataFrame()

    ref_date = reference_date or work['sale_date'].max()

    grp = work.groupby('product').agg(
        last_sale=('sale_date', 'max'),
        frequency=('sale_date', 'count'),
        monetary=('revenue', 'sum'),
    ).reset_index()

    grp['recency'] = (ref_date - grp['last_sale']).dt.days

    # Квантильный скоринг 1-5 (5 = лучше). Для recency ниже = лучше, поэтому
    # инвертируем через labels в обратном порядке.
    def safe_qcut(series, ascending_is_better):
        try:
            labels = [1, 2, 3, 4, 5] if ascending_is_better else [5, 4, 3, 2, 1]
            return pd.qcut(series.rank(method='first'), 5, labels=labels).astype(int)
        except ValueError:
            # недостаточно уникальных значений для 5 квантилей
            return pd.Series([3] * len(series), index=series.index)

    grp['r_score'] = safe_qcut(grp['recency'], ascending_is_better=False)
    grp['f_score'] = safe_qcut(grp['frequency'], ascending_is_better=True)
    grp['m_score'] = safe_qcut(grp['monetary'], ascending_is_better=True)

    def segment(row):
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'бестселлер'
        if m >= 4 and f <= 2:
            return 'скрытый потенциал'
        if r <= 2 and f >= 4:
            return 'угасающий хит'
        if r <= 2 and f <= 2 and m <= 2:
            return 'мёртвый груз'
        return 'стабильный'

    grp['segment'] = grp.apply(segment, axis=1)

    return grp[['product', 'recency', 'frequency', 'monetary',
                'r_score', 'f_score', 'm_score', 'segment']].sort_values(
        'monetary', ascending=False).reset_index(drop=True)
