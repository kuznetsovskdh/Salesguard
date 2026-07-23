"""Локальный LLM-провайдер через Ollama.

ЗАГЛУШКА: интерфейс реализован по контракту LLMProvider, но требует
локально установленного и запущенного Ollama с загруженной моделью, чего
пока нет в этом окружении (Mac Air, ограниченная память). Используется для
демонстрации провайдер-агностичной архитектуры и как путь к продакшену,
где приватность данных (никаких данных о продажах во внешнем API) важнее
скорости разработки.

Рекомендация по модели при реальном разворачивании (см. отчёт сессии):
- qwen2.5:7b - основной вариант, баланс качества/ресурсов (~4.5 ГБ на диске,
  6-8 ГБ ОЗУ при работе)
- llama3.2:3b - облегчённый вариант при нехватке памяти (~2 ГБ на диске,
  ~4 ГБ ОЗУ)

Установка (когда потребуется):
  brew install ollama
  ollama serve &
  ollama pull qwen2.5:7b
"""
import json
import requests

from analytics.llm_providers.base import LLMProvider

OLLAMA_DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_DEFAULT_URL):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, timeout: int = 30) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(self.base_url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data["response"].strip()
