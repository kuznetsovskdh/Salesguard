"""Внешний LLM-провайдер через Groq API (OpenAI-совместимый формат).

ВАЖНО: данные отчёта (метрики продаж) отправляются на серверы Groq для
обработки. Это временное решение для разработки/тестирования - PRD модуля 5
заявляет приватность данных как ценностное предложение, что подразумевает
локальный инференс (LocalOllamaProvider) для продакшена. Использование
внешнего API - явный компромисс на время, пока локальная модель не
установлена/не протестирована."""
import os
import json
import requests
from dotenv import load_dotenv

from analytics.llm_providers.base import LLMProvider

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY не задан. Передайте api_key явно или "
                "установите переменную окружения GROQ_API_KEY."
            )
        self.model = model

    def generate(self, prompt: str, timeout: int = 30) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        response = requests.post(
            GROQ_API_URL, headers=headers, json=payload, timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
