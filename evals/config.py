"""Typed view of evals/models.toml (models, prices, free-tier quotas, held-out contamination)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

MODELS_FILE = Path(__file__).with_name("models.toml")

Provider = Literal["mock", "gemini", "groq"]


@dataclass(frozen=True, slots=True)
class FreeTier:
    rpm: int
    rpd: int
    tpm: int
    tpd: int  # 0 = no known daily token cap
    daily_reset_utc_hour: int
    quota_source: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    provider: Provider
    model: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    prices_as_of: str
    price_source: str
    api_key_env: str | None = None
    price_valid_until: str | None = None
    price_after: str | None = None
    est_output_tokens_per_call: int = 500
    est_seconds_per_call: float = 2.5
    replaces: str | None = None
    notes: str = ""
    free_tier: FreeTier | None = None

    @property
    def is_llm(self) -> bool:
        return self.provider != "mock"

    @property
    def slug(self) -> str:
        return "".join(c if c.isalnum() or c in "-." else "-" for c in self.id).strip("-")

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens / 1e6 * self.input_usd_per_mtok + output_tokens / 1e6 * self.output_usd_per_mtok

    def public(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "input_usd_per_mtok": self.input_usd_per_mtok,
            "output_usd_per_mtok": self.output_usd_per_mtok,
            "prices_as_of": self.prices_as_of,
            "price_source": self.price_source,
            "price_valid_until": self.price_valid_until,
            "price_after": self.price_after,
            "replaces": self.replaces,
            "notes": self.notes,
        }
        if self.free_tier:
            ft = self.free_tier
            out["free_tier"] = {"rpm": ft.rpm, "rpd": ft.rpd, "tpm": ft.tpm, "tpd": ft.tpd, "quota_source": ft.quota_source}
        return out


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    models: dict[str, ModelSpec]
    default_models: list[str]
    max_fallback_rate: float
    contaminated_heldout: dict[str, str]
    est_calls_per_case: float
    est_input_tokens_per_call: int
    source: Path = field(default=MODELS_FILE)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self.models[model_id]
        except KeyError:
            raise SystemExit(f"unknown model id {model_id!r}; known: {', '.join(self.models)} (evals/models.toml)") from None


def load_config(path: Path = MODELS_FILE) -> BenchmarkConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    models: dict[str, ModelSpec] = {}
    for m in raw.get("model", []):
        ft_raw = m.get("free_tier")
        free_tier = (
            FreeTier(
                rpm=int(ft_raw["rpm"]),
                rpd=int(ft_raw["rpd"]),
                tpm=int(ft_raw.get("tpm", 0)),
                tpd=int(ft_raw.get("tpd", 0)),
                daily_reset_utc_hour=int(ft_raw.get("daily_reset_utc_hour", 0)),
                quota_source=str(ft_raw.get("quota_source", "")),
            )
            if ft_raw
            else None
        )
        provider = str(m["provider"])
        if provider not in ("mock", "gemini", "groq"):
            raise ValueError(f"{path}: model {m.get('id')!r} has unknown provider {provider!r}")
        spec = ModelSpec(
            id=str(m["id"]),
            provider=provider,  # type: ignore[arg-type]
            model=str(m["model"]),
            input_usd_per_mtok=float(m["input_usd_per_mtok"]),
            output_usd_per_mtok=float(m["output_usd_per_mtok"]),
            prices_as_of=str(m["prices_as_of"]),
            price_source=str(m.get("price_source", "")),
            api_key_env=m.get("api_key_env"),
            price_valid_until=m.get("price_valid_until"),
            price_after=m.get("price_after"),
            est_output_tokens_per_call=int(m.get("est_output_tokens_per_call", 500)),
            est_seconds_per_call=float(m.get("est_seconds_per_call", 2.5)),
            replaces=m.get("replaces"),
            notes=str(m.get("notes", "")),
            free_tier=free_tier,
        )
        if spec.is_llm and not spec.api_key_env:
            raise ValueError(f"{path}: LLM model {spec.id!r} needs api_key_env")
        models[spec.id] = spec
    bench = raw.get("benchmark", {})
    est = raw.get("estimate_defaults", {})
    return BenchmarkConfig(
        models=models,
        default_models=list(bench.get("default_models", list(models))),
        max_fallback_rate=float(bench.get("max_fallback_rate", 0.02)),
        contaminated_heldout=dict(raw.get("heldout", {}).get("contaminated", {})),
        est_calls_per_case=float(est.get("calls_per_case", 4.5)),
        est_input_tokens_per_call=int(est.get("input_tokens_per_call", 2000)),
        source=path,
    )
