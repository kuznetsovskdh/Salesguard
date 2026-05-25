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
