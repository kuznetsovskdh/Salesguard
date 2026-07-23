"""Единая точка сборки контекста метрик для AI Insights.
Используется и шаблонным генератором (5.1), и LLM-версией (5.4+) - один и
тот же словарь передаётся в оба, чтобы не дублировать сборку данных.
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
    """
    context = {
        'total_revenue': float(df['revenue'].sum()) if 'revenue' in df.columns else None,
        'total_rows': len(df),
        'unique_products': int(df['product'].nunique()) if 'product' in df.columns else None,
    }

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

    if cohort_df is not None and not cohort_df.empty:
        context['cohorts'] = {
            'available': True,
            'cohort_count': len(cohort_df),
        }
    else:
        context['cohorts'] = {'available': False}

    return context


def serialize_context_for_prompt(context: dict) -> str:
    """Сериализует словарь метрик в компактный текстовый блок для промпта LLM.

    Правила форматирования (чтобы модель не путалась в масштабе цифр):
    - Все денежные суммы округлены до целых рублей, с указанием "руб."
    - Проценты округлены до 1 знака после запятой, с явным знаком "%"
    - Отсутствующие модули помечены явной строкой "недоступно" - модель не
      должна додумывать то, чего нет
    - Обязательные поля (total_revenue, unique_products) идут первыми,
      опциональные (ABC/RFM/Margin/Cohorts) - по мере доступности
    """
    lines = []

    lines.append(f"Общая выручка: {context.get('total_revenue', 0):.0f} руб.")
    lines.append(f"Количество строк в отчёте: {context.get('total_rows', 0)}")
    if context.get('unique_products') is not None:
        lines.append(f"Уникальных товаров: {context['unique_products']}")

    abc = context.get('abc', {})
    if abc.get('available'):
        counts = abc.get('group_counts', {})
        lines.append(
            f"ABC-анализ: группа A - {counts.get('A', 0)} товаров, "
            f"группа B - {counts.get('B', 0)} товаров, "
            f"группа C - {counts.get('C', 0)} товаров. "
            f"Лидер продаж: {abc.get('top_product', '—')} "
            f"({abc.get('top_product_revenue', 0):.0f} руб. выручки)."
        )
    else:
        lines.append("ABC-анализ: недоступно.")

    rfm = context.get('rfm', {})
    if rfm.get('available'):
        seg = rfm.get('segment_counts', {})
        seg_str = ", ".join(f"{k}: {v}" for k, v in seg.items())
        lines.append(
            f"RFM-сегментация ({rfm.get('total_skus', 0)} SKU): {seg_str}. "
            f"Доля мёртвого груза: {rfm.get('dead_stock_pct', 0):.1f}%, "
            f"блокирует {rfm.get('dead_stock_revenue_blocked', 0):.0f} руб. в обороте."
        )
    else:
        lines.append("RFM-сегментация: недоступно.")

    margin = context.get('margin', {})
    if margin.get('available'):
        l1 = margin.get('avg_margin_l1_pct', 0)
        if margin.get('has_layer2'):
            l2 = margin.get('avg_margin_l2_pct', 0)
            lines.append(
                f"Маржинальность: средняя маржа после комиссии маркетплейса "
                f"(Слой 1) - {l1:.1f}%. Средняя чистая маржа после "
                f"себестоимости (Слой 2) - {l2:.1f}%."
            )
        else:
            lines.append(
                f"Маржинальность: средняя маржа после комиссии маркетплейса "
                f"(Слой 1) - {l1:.1f}%. Данные о себестоимости (Слой 2) не загружены."
            )
    else:
        lines.append("Маржинальность: недоступно.")

    cohorts = context.get('cohorts', {})
    if cohorts.get('available'):
        lines.append(f"Товарные когорты: обнаружено {cohorts.get('cohort_count', 0)} когорт(ы).")
    else:
        lines.append("Товарные когорты: недоступно.")

    return "\n".join(lines)
