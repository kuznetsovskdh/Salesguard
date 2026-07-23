"""Провайдер-агностичный интерфейс для LLM-инференса.

Позволяет переключаться между локальными моделями (Ollama) и внешними API
(Groq и т.п.) через единый контракт, не переписывая вызывающий код."""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, timeout: int = 30) -> str:
        """Генерирует текст по промпту. Должен бросать исключение при
        ошибке/таймауте - вызывающий код (generate_ai_insights) отвечает
        за fallback на шаблонный генератор."""
        raise NotImplementedError
