import pandas as pd

def seasonality(df: pd.DataFrame) -> pd.DataFrame:
    if 'month' not in df.columns:
        return pd.DataFrame()
    return (df.groupby(['year', 'month'])['revenue']
              .sum().reset_index()
              .sort_values(['year', 'month']))


def product_lifecycle_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Когорта товара по месяцу первой продажи + retention по месяцам.

    Для каждого SKU определяется месяц первой продажи (когорта). Затем для
    каждой когорты вычисляется % SKU, оставшихся "живыми" (была хотя бы одна
    продажа) в M+0, M+1, M+2... относительно месяца когорты.

    Возвращает heatmap-таблицу: cohort_month x month_offset -> retention_pct.
    """
    if 'product' not in df.columns or df['product'].isna().all():
        return pd.DataFrame()
    if 'sale_date' not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    if 'is_return' in work.columns:
        work = work[~work['is_return']]

    if work.empty:
        return pd.DataFrame()

    work['sale_month'] = work['sale_date'].dt.to_period('M')

    # Когорта каждого SKU = месяц первой продажи
    cohort_map = work.groupby('product')['sale_month'].min()
    work['cohort_month'] = work['product'].map(cohort_map)

    # Для каждой пары (product, sale_month) считаем месяц offset от когорты
    active = work[['product', 'sale_month', 'cohort_month']].drop_duplicates()
    active['month_offset'] = (
        (active['sale_month'].dt.year - active['cohort_month'].dt.year) * 12 +
        (active['sale_month'].dt.month - active['cohort_month'].dt.month)
    )

    cohort_sizes = cohort_map.value_counts().rename('cohort_size')

    retention = active.groupby(['cohort_month', 'month_offset'])['product'].nunique()
    retention = retention.rename('active_count').reset_index()
    retention = retention.merge(
        cohort_sizes, left_on='cohort_month', right_index=True
    )
    retention['retention_pct'] = (
        retention['active_count'] / retention['cohort_size'] * 100
    ).round(1)

    pivot = retention.pivot(
        index='cohort_month', columns='month_offset', values='retention_pct'
    ).sort_index()

    # Заполняем нулями пропущенные месяцы В ПРЕДЕЛАХ наблюдаемого диапазона
    # каждой когорты (churn может быть 0%, а не отсутствием данных - иначе
    # необратимый churn "теряется" как NaN вместо честного 0%). Периоды ПОСЛЕ
    # последнего наблюдаемого месяца в датасете остаются NaN - там действительно
    # нет данных (а не 0% продаж).
    if not pivot.empty:
        max_offset_overall = pivot.columns.max()
        last_data_month = work['sale_month'].max()
        for cohort_month in pivot.index:
            cohort_period = pd.Period(cohort_month, freq='M')
            max_observable_offset = (
                (last_data_month.year - cohort_period.year) * 12 +
                (last_data_month.month - cohort_period.month)
            )
            for offset in range(0, min(max_offset_overall, max_observable_offset) + 1):
                if offset not in pivot.columns:
                    pivot[offset] = None
                if pd.isna(pivot.loc[cohort_month, offset]):
                    pivot.loc[cohort_month, offset] = 0.0
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot.index = pivot.index.astype(str)
    pivot.columns = [f'M+{c}' for c in pivot.columns]

    return pivot.reset_index().rename(columns={'cohort_month': 'cohort'})
