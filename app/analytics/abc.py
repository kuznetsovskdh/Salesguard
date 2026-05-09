import pandas as pd

def abc_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if 'product' not in df.columns or df['product'].isna().all():
        return pd.DataFrame()
    grp = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
    total = grp.sum()
    cumshare = grp.cumsum() / total
    def group(x):
        if x <= 0.8:  return 'A'
        if x <= 0.95: return 'B'
        return 'C'
    return pd.DataFrame({
        'product': grp.index,
        'revenue': grp.values,
        'cumshare': cumshare.values,
        'abc_group': cumshare.apply(group).values
    })
