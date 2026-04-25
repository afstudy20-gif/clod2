"""Tavily search provider.

Tavily is not a chat-completion LLM. This adapter exposes Tavily search modes
through the same provider interface so CClaude can use it from the model picker.
"""
from typing import Iterator

import requests

from .base import BaseProvider, Message, ToolCall


class TavilyProvider(BaseProvider):
    name = "Tavily Search"
    SEARCH_URL = "https://api.tavily.com/search"

    DEFAULT_MODELS = {
        "tavily-search": "tavily-search",
        "tavily-answer-basic": "tavily-answer-basic",
        "tavily-answer-advanced": "tavily-answer-advanced",
    }

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or "tavily-answer-basic")

    @classmethod
    def fetch_available_models(cls, api_key: str) -> list[str]:
        return list(cls.DEFAULT_MODELS.values())

    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        query = _latest_user_query(messages)
        if not query:
            yield "Ask a search question to use Tavily."
            return

        payload = {
            "query": query,
            "max_results": 5,
            "search_depth": "advanced" if self.model == "tavily-answer-advanced" else "basic",
            "include_answer": self.model != "tavily-search",
            "include_raw_content": False,
        }
        resp = requests.post(
            self.SEARCH_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("answer")
        if answer:
            yield answer.strip()
            yield "\n\n"

        results = data.get("results", [])
        if not results:
            yield "No Tavily results found."
            return

        yield "Sources:\n"
        for i, item in enumerate(results, 1):
            title = item.get("title") or item.get("url") or f"Result {i}"
            url = item.get("url", "")
            content = (item.get("content") or "").strip()
            line = f"{i}. [{title}]({url})"
            if content:
                line += f" - {content[:220]}"
            yield line + "\n"


def _latest_user_query(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return str(msg.content).strip()
    return ""
