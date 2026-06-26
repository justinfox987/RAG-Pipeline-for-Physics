from __future__ import annotations
from typing import Protocol, Any


class LLMProvider(Protocol):
    def embed_texts(
        self, texts: list[str], model: str
    ) -> tuple[list[list[float]], dict[str, int]]:
        """
        Returns: (embeddings, usage)
        usage should include at least: prompt_tokens (and optionally completion_tokens)
        """

    def transcribe_image(
        self,
        image_data_uri: str,
        prompt: str,
        model: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> tuple[str, dict[str, int]]:
        """Returns: (text, usage)."""

    def reason(
        self,
        *,
        system_prompt: str,
        user_messages: list[dict[str, Any]],
        model: str,
        temperature: float | None,
        max_tokens: int,
        timeout: int | None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Returns: (text, usage)."""

    def get_budget_info(self) -> dict | None:
        """Optional; return None if not supported."""
        ...

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """Estimated USD cost, or None if pricing for this model is unknown."""
