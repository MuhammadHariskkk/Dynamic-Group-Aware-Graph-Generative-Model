"""Export schema definitions for ETH/UCY visualization workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass
class ExportRow:
    """Single trajectory point record."""

    scene_id: str
    sequence_id: int
    agent_id: int
    time_index: int
    frame_index: int
    phase: str
    mode_id: int
    mode_probability: float
    x: float
    y: float
    vx: float
    vy: float
    heading_rad: float
    group_id: int
    group_size: int
    intra_group_consistency: float
    inter_group_conflict: float
    group_density: float
    is_prediction: int
    is_deterministic: int


EXPORT_COLUMNS: list[str] = list(ExportRow.__annotations__.keys())


def rows_to_dataframe(rows: list[ExportRow]) -> pd.DataFrame:
    """Convert rows to deterministic, schema-fixed DataFrame."""
    if not rows:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    df = pd.DataFrame([asdict(r) for r in rows], columns=EXPORT_COLUMNS)
    df = df.sort_values(
        by=["scene_id", "sequence_id", "agent_id", "time_index", "phase", "mode_id"],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)
    return df


def validate_export_dataframe(df: pd.DataFrame) -> None:
    """Validate required export columns exist."""
    missing = [c for c in EXPORT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing export columns: {missing}")
