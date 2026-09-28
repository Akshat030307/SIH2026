"""Small shared helpers."""

from __future__ import annotations

import math

import pandas as pd


def _cell(v) -> str:
    if isinstance(v, float):
        return "–" if math.isnan(v) else f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 1e5 else f"{v:.3g}"
    return str(v)


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """DataFrame → GitHub markdown table (no tabulate dependency)."""
    d = df.reset_index() if index else df
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)
