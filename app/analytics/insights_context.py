"""Единая точка сборки контекста метрик для AI Insights.
Используется и шаблонным генератором (5.1), и будущей LLM-версией (5.4+) -
один и тот же словарь передаётся в оба, чтобы не дублировать сборку данных.
"""
import pandas as pd


def build_metrics_context(
    df: pd.DataFrame,
    abc_df: pd.DataFrame = None,
    rfm_df: pd.DataFrame = None,
    margin_df: pd.DataFrame = None,
    cohort_df: pd.DataFrame = None,
) -> dict:
    """Собирает агрегированные метрики из уже посчитанных модулей 2-4 в единый
    словарь. Каждый модуль опционален - если не передан (None или пустой),
    соответствующий раздел контекста помечается как недоступный, а не падает.

    Возвращаемый словарь - "контракт" для генераторов инсайтов: и шаблонного
    (analytics/insights_template.py), и будущего LLM-based.
    """
    context = {
        'total_revenue': float(df['revenue'].sum()) if 'revenue' in df.columns else None,
        'total_rows': len(df),
        'unique_products': int(df['product'].nunique()) if 'product' in df.columns else None,
    }

    # ABC/Pareto
    if abc_df is not None and not abc_df.empty:
        group_counts = abc_df['abc_group'].value_counts().to_dict()
        context['abc'] = {
            'available': True,
            'group_counts': {k: int(v) for k, v in group_counts.items()},
            'top_product': str(abc_df.iloc[0]['product']),
            'top_product_revenue': float(abc_df.iloc[0]['revenue']),
        }
    else:
        context['abc'] = {'available': False}

    # RFM
    if rfm_df is not None and not rfm_df.empty:
        seg_counts = rfm_df['segment'].value_counts().to_dict()
        total = len(rfm_df)
        dead_stock = rfm_df[rfm_df['segment'] == 'мёртвый груз']
        context['rfm'] = {
            'available': True,
            'segment_counts': {k: int(v) for k, v in seg_counts.items()},
            'total_skus': total,
            'dead_stock_pct': round(len(dead_stock) / total * 100, 1) if total else 0,
            'dead_stock_revenue_blocked': float(dead_stock['monetary'].sum()) if not dead_stock.empty else 0,
        }
    else:
        context['rfm'] = {'available': False}

    # Margin
    if margin_df is not None and not margin_df.empty:
        context['margin'] = {
            'available': True,
            'avg_margin_l1_pct': round(float(margin_df['margin_l1_pct'].mean()) * 100, 1),
            'has_layer2': bool(margin_df['margin_l2'].notna().any()) if 'margin_l2' in margin_df.columns else False,
        }
        if context['margin']['has_layer2']:
            l2 = margin_df[margin_df['margin_l2'].notna()]
            context['margin']['avg_margin_l2_pct'] = round(float(l2['margin_l2_pct'].mean()) * 100, 1)
    else:
        context['margin'] = {'available': False}

    # Lifecycle cohorts
    if cohort_df is not None and not cohort_df.empty:
        context['cohorts'] = {
            'available': True,
            'cohort_count': len(cohort_df),
        }
    else:
        context['cohorts'] = {'available': False}

    return context
