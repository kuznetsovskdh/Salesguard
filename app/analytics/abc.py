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


def pareto_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Парето-профиль товаров: ранг, выручка, накопленная доля.
    Без буквенных ABC-групп — сырые данные для графика/дальнейшей интерпретации."""
    if 'product' not in df.columns or df['product'].isna().all():
        return pd.DataFrame()
    grp = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
    total = grp.sum()
    cumshare = grp.cumsum() / total
    rank = range(1, len(grp) + 1)
    return pd.DataFrame({
        'rank': rank,
        'product': grp.index,
        'revenue': grp.values,
        'cumshare': cumshare.values
    })
