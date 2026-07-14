import json
import re
import urllib.request


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
            # input only; no output tokens
            "cohere-embed-v4":       (0.12,  0.00),
        }
        self._pricing_cache: dict[str, tuple[float, float]] = {}
        self._models_cache: list[str] = []
        self._pricing_loaded = False
        self._models_loaded = False
        self._pricing_url = "https://cborg.lbl.gov/models/"

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

        # Stream the response so we always get content even if max_tokens is hit.
        # Without streaming, CBorg/Gemini returns content=None on truncation.
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        chunks = []
        finish_reason = None
        usage_obj = None
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    chunks.append(delta.content)
                fr = chunk.choices[0].finish_reason
                if fr:
                    finish_reason = fr
            if getattr(chunk, "usage", None):
                usage_obj = chunk.usage

        text = "".join(chunks).strip()

        # Fall back to reasoning_content for o-series / thinking models
        if not text and finish_reason != "length":
            text = ""  # nothing recoverable

        if not text:
            raise ValueError(
                f"Empty response from model (finish_reason={finish_reason!r}). "
                f"The model produced no output — try again or switch models."
            )

        if finish_reason == "length":
            text += "\n\n*(Note: response was cut off at the output token limit.)*"

        return text, self._usage_dict_from_obj(usage_obj)

    # ── Streaming reasoning ────────────────────────────────────────────
    def stream_reason(
        self,
        *,
        system_prompt: str,
        user_messages: list,
        model: str,
        temperature: float | None,
        max_tokens: int,
        timeout: int | None = None,
        extra_kwargs: dict | None = None,
    ):
        """Generator: yields (text_chunk, None) for each content chunk,
        then ('', usage_dict) as the final item so callers capture token counts."""
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *user_messages,
            ],
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None and not self._omit_temperature(model):
            kwargs["temperature"] = temperature
        if timeout is not None:
            kwargs["timeout"] = timeout
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        usage_obj = None
        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content, None
            if getattr(chunk, "usage", None):
                usage_obj = chunk.usage

        yield "", self._usage_dict_from_obj(usage_obj)

    # ── Budget ─────────────────────────────────────────────────────────

    def estimate_cost(self, model, input_tokens, output_tokens):
        """Estimated USD cost, or None if pricing for this model is unknown."""
        if not self._pricing_loaded:
            self._load_pricing_from_web()
        price = self._pricing_cache.get(model) or self._PRICING.get(model)
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

    def list_models(self) -> list[str]:
        if not self._models_loaded:
            self._load_models_from_api()
            self._load_pricing_from_web()
        models = self._models_cache or []
        if not models:
            if not self._pricing_loaded:
                self._load_pricing_from_web()
            models = list(self._pricing_cache.keys())
        if not models:
            models = list(self._PRICING.keys())
        if not models:
            return []
        return sorted(set(models))

    # ── Provider-internal helpers ──────────────────────────────────────
    def _omit_temperature(self, model: str) -> bool:
        """CBorg/OpenAI-compatible: some models reject the temperature param."""
        return "claude" in model.lower()

    def _load_models_from_api(self):
        self._models_loaded = True
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {}
            for path in ("/models", "/v1/models"):
                try:
                    req = urllib.request.Request(f"{self.base_url}{path}", headers=headers)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        payload = json.loads(resp.read().decode())
                    if payload:
                        break
                except Exception:
                    continue
            data = payload.get("data", payload.get("models", []))
            self._models_cache = [m.get("id") or m.get("model") for m in data if (m.get("id") or m.get("model"))]
        except Exception:
            self._models_cache = []

    def _load_pricing_from_web(self):
        if self._pricing_loaded:
            return
        self._pricing_loaded = True
        try:
            req = urllib.request.Request(self._pricing_url)
            with urllib.request.urlopen(req, timeout=12) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            self._pricing_cache = self._parse_pricing_html(html)
        except Exception:
            self._pricing_cache = {}

    def _parse_pricing_html(self, html: str) -> dict[str, tuple[float, float]]:
        pricing: dict[str, tuple[float, float]] = {}
        # Pattern: <h4 id=model-name>model-name</h4> ... Cost per 1M Tokens (Input): $X ... (Output): $Y
        sections = re.split(r"<h4\\s+id=", html, flags=re.I)
        for sec in sections[1:]:
            head_end = sec.find("</h4>")
            if head_end == -1:
                continue
            head_raw = sec[:head_end]
            model_id = re.sub(r"[\\\"'>].*", "", head_raw).strip()
            if not model_id:
                continue
            input_match = re.search(
                r"Cost per 1M Tokens \\(Input\\)</strong>:\\s*\\$([0-9]+(?:\\.[0-9]+)?)",
                sec,
                flags=re.I,
            )
            output_match = re.search(
                r"Cost per 1M Tokens \\(Output\\)</strong>:\\s*\\$([0-9]+(?:\\.[0-9]+)?)",
                sec,
                flags=re.I,
            )
            if input_match and output_match:
                pricing[model_id] = (float(input_match.group(1)), float(output_match.group(1)))
            elif input_match:
                pricing[model_id] = (float(input_match.group(1)), 0.0)
        return pricing

    @staticmethod
    def _usage_dict(response) -> dict[str, int]:
        return CBorgProvider._usage_dict_from_obj(getattr(response, "usage", None))

    @staticmethod
    def _usage_dict_from_obj(u) -> dict[str, int]:
        if u is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        }
