import urllib.request
import json




class CBorgProvider:
    """OpenAI-compatible provider for CBorg. Reference implementation of the
    LLMProvider protocol (see base.py). All calls go through an OpenAI-style
    client; response-shape quirks (reasoning_content, temperature handling)
    are absorbed here so the scripts stay provider-agnostic."""

    def __init__(self, api_key: str, base_url: str):
        import openai
        self.api_key = api_key
        self.base_url = base_url
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._PRICING = {
                "gpt-5.1":               (1.25, 10.00),
                "gpt-5.4-pro":           (2.50, 15.00),
                "claude-sonnet-4-6":     (3.00, 15.00),
                "claude-sonnet-high":    (3.00, 15.00),
                "claude-opus-4-6":       (5.00, 25.00),
                "claude-haiku-4-5":      (1.00,  5.00),
                "gemini-2.0-flash":      (0.10,  0.40),
                "gemini-2.5-flash":      (0.30,  2.50),
                "gemini-3.1-flash-lite": (0.25,  1.50),
            }

    # ── Embedding ──────────────────────────────────────────────────────
    def embed_texts(
        self, texts: list[str], model: str
    ) -> tuple[list[list[float]], dict[str, int]]:
        response = self.client.embeddings.create(model=model, input=texts)
        ordered = sorted(response.data, key=lambda d: d.index)
        embeddings = [d.embedding for d in ordered]
        usage = {}
        u = getattr(response, "usage", None)
        if u is not None:
            usage["prompt_tokens"] = getattr(u, "prompt_tokens", 0) or 0
        return embeddings, usage

    # ── Vision transcription ───────────────────────────────────────────
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
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ]}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        message = response.choices[0].message
        text = (message.content or "").strip()
        if not text:
            text = (getattr(message, "reasoning_content", "") or "").strip()
        usage = self._usage_dict(response)
        usage["finish_reason"] = response.choices[0].finish_reason
        return text, usage

    # ── Reasoning ──────────────────────────────────────────────────────
    def reason(
        self,
        *,
        system_prompt: str,
        user_messages: list,
        model: str,
        temperature: float | None,
        max_tokens: int,
        timeout: int | None,
        extra_kwargs: dict | None = None,
    ) -> tuple[str, dict[str, int]]:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *user_messages,
            ],
            "max_tokens": max_tokens,
        }
        # OpenAI-compatible quirk: reasoning models reject temperature.
        # CBorg routes claude-* without temperature; everything else takes it.
        if temperature is not None and not self._omit_temperature(model):
            kwargs["temperature"] = temperature
        if timeout is not None:
            kwargs["timeout"] = timeout
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        text = (message.content or "").strip()
        if not text:
            text = (getattr(message, "reasoning_content", None)
                    or f"[Empty response — raw: {message}]")
        return text, self._usage_dict(response)

    # ── Budget ─────────────────────────────────────────────────────────

    def estimate_cost(self, model, input_tokens, output_tokens):
        """Estimated USD cost, or None if pricing for this model is unknown."""
        price = self._PRICING.get(model)
        if price is None:
            return None
        return (input_tokens / 1_000_000 * price[0]) + (output_tokens / 1_000_000 * price[1])

    def get_budget_info(self) -> dict | None:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/user/info",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            info = data.get("user_info", data)
            return {
                "spend": info.get("spend", data.get("spend")),
                "max_budget": info.get("max_budget", data.get("max_budget")),
                "budget_reset_at": info.get("budget_reset_at", data.get("budget_reset_at")),
                "_raw": data,
            }
        except Exception:
            return None

    # ── Provider-internal helpers ──────────────────────────────────────
    def _omit_temperature(self, model: str) -> bool:
        """CBorg/OpenAI-compatible: some models reject the temperature param."""
        return "claude" in model.lower()

    @staticmethod
    def _usage_dict(response) -> dict[str, int]:
        u = getattr(response, "usage", None)
        if u is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        }