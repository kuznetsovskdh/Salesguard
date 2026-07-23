"""Сборка и генерация AI Insights через LLM с fallback на шаблонный генератор.

generate_ai_insights() - единая точка входа для app.py: собирает промпт из
контекста метрик, пытается вызвать LLM-провайдер, при любой ошибке/таймауте
прозрачно переключается на шаблонный генератор (analytics/insights_template.py)
без ошибки для пользователя."""
import logging

from analytics.insights_context import serialize_context_for_prompt
from analytics.insights_prompt import build_prompt
from analytics.insights_template import generate_template_insights

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 15


def generate_ai_insights(context: dict, provider=None) -> str:
    """Генерирует текст-интерпретацию метрик.

    Если provider передан и генерация через него успешна - возвращает
    LLM-текст. При любой ошибке (нет API-ключа, таймаут, сетевая ошибка,
    провайдер не передан) - тихо откатывается на шаблонный генератор,
    не показывая пользователю техническую ошибку.
    """
    if provider is not None:
        try:
            context_text = serialize_context_for_prompt(context)
            prompt = build_prompt(context_text)
            return provider.generate(prompt, timeout=LLM_TIMEOUT_SECONDS)
        except Exception as e:
            logger.warning(
                "LLM provider failed, falling back to template insights: %s", e
            )

    return generate_template_insights(context)
