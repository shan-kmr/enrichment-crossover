"""OpenAI API client with exponential backoff retry, rate limiting, and disk-based response caching."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from openai import OpenAI, APIError, RateLimitError, APITimeoutError


class CachedLLMClient:
    def __init__(
        self,
        cache_dir: str | Path = "results/cache",
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
        request_timeout: float = 60.0,
        min_request_interval: float = 0.1,
    ):
        self.client = OpenAI()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.request_timeout = request_timeout
        self.min_request_interval = min_request_interval
        self._last_request_time = 0.0

    def _cache_key(self, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _legacy_cache_key(self, model: str, messages: list[dict], temperature: float) -> str:
        payload = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _write_cache(self, key: str, data: dict) -> None:
        self._cache_path(key).write_text(json.dumps(data))

    def _rate_limit_wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict:
        """Send a chat completion request with caching and retry.

        Returns dict with keys: content, model, usage, cached.
        """
        cache_key = self._cache_key(model, messages, temperature, max_tokens)
        cached = self._read_cache(cache_key)
        if cached is not None:
            cached["cached"] = True
            return cached

        legacy_key = self._legacy_cache_key(model, messages, temperature)
        cached = self._read_cache(legacy_key)
        if cached is not None and cached.get("content"):
            self._write_cache(cache_key, cached)
            cached["cached"] = True
            return cached

        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()
                self._last_request_time = time.time()

                is_reasoning = "gpt-5" in model or "o3" in model or "o4" in model
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "timeout": self.request_timeout,
                    "max_completion_tokens" if is_reasoning else "max_tokens": max_tokens,
                }
                if is_reasoning:
                    kwargs["reasoning_effort"] = "low"
                else:
                    kwargs["temperature"] = temperature
                response = self.client.chat.completions.create(**kwargs)

                result = {
                    "content": response.choices[0].message.content,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    "cached": False,
                }
                self._write_cache(cache_key, result)
                return result

            except RateLimitError:
                delay = self.retry_base_delay * (2 ** attempt)
                print(f"  Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)

            except (APITimeoutError, APIError) as e:
                delay = self.retry_base_delay * (2 ** attempt)
                print(f"  API error: {e}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)

        raise RuntimeError(f"Failed after {self.max_retries} retries for model {model}")
