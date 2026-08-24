"""Small helpers for immutable reviewed experiment outputs."""

from pathlib import Path

import pandas as pd


def verify_or_create_tsv(result: pd.DataFrame, path: Path) -> None:
    """Create an initial check table, or verify an existing one without rewriting it."""
    expected = result.to_csv(sep="\t", index=False, lineterminator="\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(
                f"Reviewed output differs from recomputation; create a new version: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")
