"""Reading-era detection with optimal change-point segmentation.

Dated books are ordered by finish date and each one becomes a one-hot vector
over categories. We then find the segmentation that minimises

    sum over segments of (squared deviation from the segment's mean mix)
    + penalty * (number of segments)

with an exact O(n^2) dynamic programme (the "optimal partitioning" problem
that PELT speeds up). With ~100 books that's instant, and it avoids a
dependency on ``ruptures``. A higher penalty gives fewer, broader eras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORIES

SHORT_NAMES = {
    "Literature": "Fiction & classics",
    "History & Biography": "History",
    "Economics & Society": "Economics & society",
    "Psychology & Behavior": "Psychology",
    "Self-help & Productivity": "Self-help",
    "Business & Entrepreneurship": "Business",
    "Religion & Ethics": "Faith & ethics",
    "Unknown": "Unclassified",
}


def segment(X: np.ndarray, penalty: float, min_size: int = 4) -> list[int]:
    """Return the end indices (exclusive) of each optimal segment of ``X``."""
    n = len(X)
    if n == 0:
        return []
    if n < 2 * min_size:
        return [n]

    csum = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])
    csq = np.concatenate([[0.0], np.cumsum((X ** 2).sum(axis=1))])

    def cost(i: int, j: int) -> float:  # SSE of rows i..j-1 around their mean
        s = csum[j] - csum[i]
        return float(csq[j] - csq[i] - (s @ s) / (j - i))

    best = np.full(n + 1, np.inf)
    best[0] = -penalty
    prev = np.zeros(n + 1, dtype=int)
    for j in range(min_size, n + 1):
        for i in range(0, j - min_size + 1):
            if np.isinf(best[i]) or (i != 0 and i < min_size):
                continue
            c = best[i] + cost(i, j) + penalty
            if c < best[j]:
                best[j], prev[j] = c, i

    ends, j = [], n
    while j > 0:
        ends.append(j)
        j = prev[j]
    return ends[::-1]


def _label(mix: pd.Series) -> str:
    top = mix.sort_values(ascending=False)
    first, second = top.index[0], top.index[1] if len(top) > 1 else None
    if top.iloc[0] >= 0.5 or second is None or top.iloc[1] < 0.2:
        return f"{SHORT_NAMES.get(first, first)} era"
    return f"{SHORT_NAMES.get(first, first)} + {SHORT_NAMES.get(second, second).lower()}"


def boundary_stability(df: pd.DataFrame, penalties=None, min_size: int = 4) -> pd.Series:
    """How robust is each era boundary? Share of penalty values that keep it.

    Boundaries are matched by the date of the first book of the new era,
    within +/- 2 books, so small shifts still count as the same boundary.
    """
    penalties = penalties if penalties is not None else np.linspace(1.0, 3.5, 26)
    dated = df[df["dated"]].sort_values(["finish_date", "id"])
    cats = sorted(set(dated["category"]))
    X = pd.get_dummies(dated["category"]).reindex(columns=cats, fill_value=0).to_numpy(float)
    hits: dict[int, int] = {}
    for p in penalties:
        for b in segment(X, p, min_size)[:-1]:
            hits[b] = hits.get(b, 0) + 1
    def support(b: int) -> float:
        return sum(v for k, v in hits.items() if abs(k - b) <= 2) / len(penalties)
    return pd.Series({b: min(1.0, support(b)) for b in hits})


def detect_eras(df: pd.DataFrame, penalty: float = 2.0, min_size: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    """Detect eras among dated books.

    Returns ``(eras, assignment)`` where ``eras`` has one row per era and
    ``assignment`` maps each dated book's index in ``df`` to an era number.
    """
    dated = df[df["dated"]].sort_values(["finish_date", "id"])
    cats = [c for c in CATEGORIES if c in set(dated["category"])] + (
        ["Unknown"] if "Unknown" in set(dated["category"]) else []
    )
    X = pd.get_dummies(dated["category"]).reindex(columns=cats, fill_value=0).to_numpy(float)
    ends = segment(X, penalty=penalty, min_size=min_size)

    rows, assignment, start = [], pd.Series(index=dated.index, dtype="Int64"), 0
    for k, end in enumerate(ends):
        part = dated.iloc[start:end]
        mix = part["category"].value_counts(normalize=True)
        rows.append(
            {
                "era": k + 1,
                "start_pos": start,
                "label": _label(mix),
                "start": part["finish_date"].min(),
                "end": part["finish_date"].max(),
                "books": len(part),
                "mix": mix.round(3).to_dict(),
                "top_category": mix.index[0],
                "examples": ", ".join(part["original_title"].head(3)),
            }
        )
        assignment.loc[part.index] = k + 1
        start = end
    return pd.DataFrame(rows), assignment
