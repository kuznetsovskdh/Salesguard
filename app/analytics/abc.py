import pandas as pd

def abc_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if 'product' not in df.columns or df['product'].isna().all():
        return pd.DataFrame()
    grp = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
    total = grp.sum()
    cumshare = grp.cumsum() / total
    # Группа определяется по накопленной доле ПРЕДЫДУЩЕЙ позиции: товар,
    # который пересекает порог 80%, сам ещё относится к A. Иначе товар с
    # 90% выручки давал cumshare=0.90 и попадал в B, а группа A оставалась
    # пустой - ровно на том файле, где группа A очевиднее всего.
    prev = cumshare.shift(1).fillna(0.0)
    def group(x):
        if x < 0.8:  return 'A'
        if x < 0.95: return 'B'
        return 'C'
    return pd.DataFrame({
        'product': grp.index,
        'revenue': grp.values,
        'cumshare': cumshare.values,
        'abc_group': prev.apply(group).values
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
