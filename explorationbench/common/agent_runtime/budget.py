from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BudgetProfile:
    """Soft resource profile used to annotate framework-track runs."""

    name: str = "c1"
    token_multiplier: float = 1.0
    max_environment_calls: int | None = None
    max_wall_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_name(cls, name: str | None) -> "BudgetProfile":
        text = (name or "c1").strip().lower()
        multipliers = {
            "c1": 1.0,
            "1x": 1.0,
            "2x": 2.0,
            "4x": 4.0,
        }
        return cls(name=text, token_multiplier=multipliers.get(text, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "token_multiplier": self.token_multiplier,
            "max_environment_calls": self.max_environment_calls,
            "max_wall_seconds": self.max_wall_seconds,
            "metadata": self.metadata,
        }
