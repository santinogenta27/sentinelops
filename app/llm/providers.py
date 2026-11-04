"""Concrete providers: Google Gemini (google-genai) and Groq (OpenAI-compatible REST)."""

from __future__ import annotations

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.llm.base import LLMClient, LLMError, LLMUnavailableError, usage_from_mapping

_RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}
_AUTH_HTTP = {401, 403}


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, *, api_key: str, model: str, rpm: int, max_retries: int, timeout_s: float) -> None:
        super().__init__(model=model, rpm=rpm, max_retries=max_retries, timeout_s=timeout_s)
        self._client = genai.Client(api_key=api_key)

    async def _complete(self, *, system: str, prompt: str, json_mode: bool) -> str:
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        if "2.5" in self.model and "flash" in self.model:
            # Deterministic, low-latency extraction: no hidden thinking budget.
            config.thinking_config = genai_types.ThinkingConfig(thinking_budget=0)
        try:
            response = await self._client.aio.models.generate_content(model=self.model, contents=prompt, config=config)
        except genai_errors.APIError as exc:
            code = int(getattr(exc, "code", 0) or 0)
            if code in _AUTH_HTTP:
                raise LLMUnavailableError(f"gemini auth failed ({code}): check GEMINI_API_KEY", status_code=code) from exc
            raise LLMError(
                f"gemini HTTP {code}: {str(exc)[:300]}", retryable=code in _RETRYABLE_HTTP, status_code=code or None
            ) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"gemini transport error: {exc}", retryable=True) from exc
        meta = response.usage_metadata
        if meta is None or meta.prompt_token_count is None:
            self._record_tokens(None, None)
        else:
            # thinking tokens are billed as output
            self._record_tokens(meta.prompt_token_count, (meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0))
        text = response.text
        if not text:
            raise LLMError("gemini returned an empty response (possibly safety-blocked)", retryable=True)
        return text


class GroqClient(LLMClient):
    provider = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, *, api_key: str, model: str, rpm: int, max_retries: int, timeout_s: float) -> None:
        super().__init__(model=model, rpm=rpm, max_retries=max_retries, timeout_s=timeout_s)
        self._api_key = api_key

    async def _complete(self, *, system: str, prompt: str, json_mode: bool) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                resp = await client.post(self.URL, json=payload, headers={"Authorization": f"Bearer {self._api_key}"})
        except httpx.TransportError as exc:
            raise LLMError(f"groq transport error: {exc}", retryable=True) from exc
        if resp.status_code in _AUTH_HTTP:
            raise LLMUnavailableError(f"groq auth failed ({resp.status_code}): check GROQ_API_KEY", status_code=resp.status_code)
        if resp.status_code >= 400:
            raise LLMError(
                f"groq HTTP {resp.status_code}: {resp.text[:300]}",
                retryable=resp.status_code in _RETRYABLE_HTTP,
                status_code=resp.status_code,
            )
        data = resp.json()
        usage = data.get("usage") if isinstance(data, dict) else None
        self._record_tokens(*usage_from_mapping(usage if isinstance(usage, dict) else None, "prompt_tokens", "completion_tokens"))
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"groq malformed response: {exc}", retryable=True) from exc
        if not isinstance(content, str) or not content:
            raise LLMError("groq returned empty content", retryable=True)
        return content
