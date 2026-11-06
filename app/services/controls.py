"""Operator controls that can change at runtime from the cockpit (persisted in ``app_settings``)."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from app.agent.autopilot import AUTOPILOT_LIMIT_USD
from app.config import Settings
from app.db.database import Database

AUTOPILOT_KEY = "autopilot"
logger = logging.getLogger("sentinelops.controls")


class AutopilotState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    max_usd: float = Field(ge=0, le=100_000)


class AutopilotUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    max_usd: float | None = Field(default=None, ge=0, le=100_000)


class RuntimeControls:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.autopilot = AutopilotState(
            enabled=settings.autopilot_default, max_usd=min(settings.autopilot_max_usd, AUTOPILOT_LIMIT_USD)
        )

    async def load(self) -> None:
        saved = await self.db.get_setting(AUTOPILOT_KEY)
        if isinstance(saved, dict):
            try:
                self.autopilot = AutopilotState.model_validate({**self.autopilot.model_dump(), **saved})
            except ValueError:
                return  # corrupt row: keep defaults
            if self.autopilot.max_usd > AUTOPILOT_LIMIT_USD:
                # A ceiling saved before the limit existed never outranks it.
                logger.warning("auto-pilot ceiling %.2f lowered to %.2f", self.autopilot.max_usd, AUTOPILOT_LIMIT_USD)
                self.autopilot = self.autopilot.model_copy(update={"max_usd": AUTOPILOT_LIMIT_USD})

    async def update_autopilot(self, update: AutopilotUpdate) -> AutopilotState:
        if update.max_usd is not None and update.max_usd > AUTOPILOT_LIMIT_USD:
            raise ValueError(f"the auto-pilot ceiling cannot exceed ${AUTOPILOT_LIMIT_USD:,.2f}")
        merged = {**self.autopilot.model_dump(), **update.model_dump(exclude_none=True)}
        self.autopilot = AutopilotState.model_validate(merged)
        await self.db.set_setting(AUTOPILOT_KEY, self.autopilot.model_dump())
        return self.autopilot
