"""Шаблонный генератор AI Insights - правило-based текст с подстановкой
чисел из build_metrics_context(). Работает как первая рабочая версия и как
fallback для будущей LLM-версии, если Ollama недоступна/медленная."""


def fmt_num(n: float) -> str:
    """Форматирует число с разделителем тысяч через пробел (без ломки текста)."""
    return f"{n:,.0f}".replace(",", " ")


def generate_template_insights(context: dict) -> str:
    """Генерирует связный текст-интерпретацию на основе словаря метрик.
    Каждый блок независим: если раздел недоступен (available=False),
    соответствующий абзац пропускается, а не выводит пустоту/ошибку."""
    parts = []

    total_revenue = context.get('total_revenue')
    unique_products = context.get('unique_products')
    if total_revenue is not None:
        parts.append(
            f"Общая выручка за период составила {fmt_num(total_revenue)} руб. "
            f"по {unique_products} товарным позициям."
        )

    abc = context.get('abc', {})
    if abc.get('available'):
        counts = abc['group_counts']
        a_count = counts.get('A', 0)
        top_product = abc.get('top_product', '—')
        top_revenue = abc.get('top_product_revenue', 0)
        parts.append(
            f"ABC-анализ: {a_count} товаров формируют основную часть выручки "
            f"(группа A). Лидер продаж — «{top_product}» с выручкой "
            f"{fmt_num(top_revenue)} руб."
        )

    rfm = context.get('rfm', {})
    if rfm.get('available'):
        dead_pct = rfm.get('dead_stock_pct', 0)
        dead_revenue = rfm.get('dead_stock_revenue_blocked', 0)
        seg_counts = rfm.get('segment_counts', {})
        bestseller_count = seg_counts.get('бестселлер', 0)
        if dead_pct > 0:
            parts.append(
                f"{dead_pct}% товаров в сегменте «мёртвый груз» — они "
                f"блокируют {fmt_num(dead_revenue)} руб. в обороте и требуют "
                f"внимания: пересмотреть цену, вывести из ассортимента или "
                f"провести распродажу."
            )
        if bestseller_count > 0:
            parts.append(
                f"{bestseller_count} товаров попали в сегмент «бестселлер» — "
                f"стабильно высокие продажи, стоит следить за наличием на складе."
            )

    margin = context.get('margin', {})
    if margin.get('available'):
        l1_pct = margin.get('avg_margin_l1_pct')
        if margin.get('has_layer2'):
            l2_pct = margin.get('avg_margin_l2_pct')
            parts.append(
                f"Средняя маржа после комиссии маркетплейса — {l1_pct}%. "
                f"С учётом себестоимости (чистая маржа) — {l2_pct}%."
            )
        else:
            parts.append(
                f"Средняя маржа после комиссии маркетплейса — {l1_pct}%. "
                f"Это маржа без учёта себестоимости — загрузите справочник "
                f"себестоимости для расчёта чистой маржи."
            )

    cohorts = context.get('cohorts', {})
    if cohorts.get('available'):
        cnt = cohorts.get('cohort_count', 0)
        if cnt > 0:
            parts.append(
                f"В данных обнаружено {cnt} товарных когорт(а) — для более "
                f"детального анализа выживаемости новых SKU нужны данные "
                f"минимум за несколько месяцев."
            )

    if not parts:
        return "Недостаточно данных для формирования инсайтов по этому отчёту."

    return " ".join(parts)
