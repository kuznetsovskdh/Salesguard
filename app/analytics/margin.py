import pandas as pd


def margin_analysis(df: pd.DataFrame, cost_df: pd.DataFrame = None) -> pd.DataFrame:
    """Маржинальный анализ в двух независимых слоях.

    Слой 1 (всегда доступен): margin_l1 = revenue - commission
        Для WB: revenue - commission
        Для Ozon: revenue - commission - logistics + points
        (formулы подтверждены сверкой на реальных данных, см. отчёт сессии)

    Слой 2 (опционально, если передан cost_df): margin_l2 = margin_l1 - unit_cost * quantity
        cost_df ожидается со столбцами: sku_or_article, cost_price

    Поля margin_l1 и margin_l2 ВСЕГДА разделены, никогда не сливаются в одну
    цифру — это принципиальное требование PRD.
    """
    if 'product' not in df.columns or df['product'].isna().all():
        return pd.DataFrame()
    if 'revenue' not in df.columns:
        return pd.DataFrame()

    # Margin L1 требует реальных данных о комиссии - если колонка отсутствует
    # или полностью пуста (отчёт не содержит этой информации, как в
    # wb_dynamics), считать margin_l1=revenue было бы вводящим в заблуждение
    # (искусственные 100%), а не честным расчётом
    has_commission_data = (
        'commission' in df.columns and not df['commission'].isna().all()
    )
    if not has_commission_data:
        return pd.DataFrame()

    work = df.copy()

    commission = work['commission'].fillna(0)
    logistics = work['logistics'].fillna(0) if 'logistics' in work.columns else 0
    points = work['points'].fillna(0) if 'points' in work.columns else 0

    # Margin L1: если есть logistics/points колонки (Ozon-паттерн) - используем
    # полную формулу, иначе (WB-паттерн) - только revenue - commission
    if 'logistics' in work.columns or 'points' in work.columns:
        work['margin_l1'] = work['revenue'] - commission - logistics + points
    else:
        work['margin_l1'] = work['revenue'] - commission

    grp = work.groupby('product').agg(
        revenue=('revenue', 'sum'),
        margin_l1=('margin_l1', 'sum'),
        quantity=('quantity', 'sum') if 'quantity' in work.columns else ('revenue', 'count'),
    ).reset_index()

    grp['margin_l1_pct'] = (grp['margin_l1'] / grp['revenue']).where(grp['revenue'] != 0)

    # Слой 2: только если передан cost_df
    if cost_df is not None and not cost_df.empty and 'sku_or_article' in cost_df.columns:
        cost_map = cost_df.set_index('sku_or_article')['cost_price'].to_dict()
        grp['unit_cost'] = grp['product'].map(cost_map)
        grp['margin_l2'] = grp.apply(
            lambda r: r['margin_l1'] - r['unit_cost'] * r['quantity']
            if pd.notna(r.get('unit_cost')) else None,
            axis=1
        )
        grp['margin_l2_pct'] = (grp['margin_l2'] / grp['revenue']).where(grp['revenue'] != 0)
    else:
        grp['unit_cost'] = None
        grp['margin_l2'] = None
        grp['margin_l2_pct'] = None

    return grp.sort_values('margin_l1', ascending=False).reset_index(drop=True)
