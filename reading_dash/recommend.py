"""Explainable 'what to read next' recommendations.

The reader profile is a recency-weighted average of the TF-IDF vectors of
books already read (half-life configurable), so recent taste counts more.
Each candidate gets a cosine score against the profile, and the two most
similar books already read are shown as the reason.

"Stretch" mode mixes in a novelty bonus for things the list rarely has:
women authors, countries not read yet and original languages not read yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recency_weights(works: pd.DataFrame, half_life_days: float = 540, undated_weight: float = 0.25) -> np.ndarray:
    dates = works["finish_date"]
    latest = dates.max()
    age = (latest - dates).dt.days.to_numpy(dtype=float)
    w = np.where(np.isnan(age), undated_weight, 0.5 ** (np.nan_to_num(age) / half_life_days))
    return w / w.sum()


def recommend(
    works: pd.DataFrame,
    candidates: pd.DataFrame,
    read_matrix,
    cand_matrix,
    mode: str = "similar",
    half_life_days: float = 540,
    n: int = 8,
) -> pd.DataFrame:
    w = recency_weights(works, half_life_days)
    profile = np.asarray(read_matrix.multiply(w[:, None]).sum(axis=0))
    sim = cosine_similarity(profile, cand_matrix).ravel()

    out = candidates.copy()
    out["similarity"] = sim

    read_titles = set(works["original_title"].str.lower()) | set(works["title"].str.lower())
    out = out[~out["title"].str.lower().isin(read_titles)]

    if mode == "stretch":
        seen_countries = set(works["author_country"])
        seen_langs = set(works["original_language"])
        women_share = (works["author_gender"] == "F").mean()
        bonus = (
            (out["author_gender"].isin(["F", "Mixed"])) * (0.5 - min(women_share, 0.5))
            + (~out["author_country"].isin(seen_countries)) * 0.25
            + (~out["original_language"].isin(seen_langs)) * 0.2
        )
        out["novelty"] = bonus
        out["score"] = 0.6 * out["similarity"] / max(out["similarity"].max(), 1e-9) + bonus
    else:
        out["novelty"] = 0.0
        out["score"] = out["similarity"]

    out = out.sort_values("score", ascending=False).head(n)

    # Explanation: which books already read is each pick closest to?
    pair = cosine_similarity(cand_matrix[out.index.to_numpy()], read_matrix)
    reasons = []
    for row in pair:
        top = np.argsort(-row)[:2]
        reasons.append([works.iloc[i]["original_title"] for i in top if row[i] > 0])
    out["because"] = reasons

    def why_new(r):
        notes = []
        if r["author_country"] not in set(works["author_country"]):
            notes.append(f"first author from {r['country_name']}")
        if r["original_language"] not in set(works["original_language"]):
            notes.append(f"first book originally in {r['original_language']}")
        if r["author_gender"] in ("F", "Mixed"):
            notes.append("woman author")
        return notes

    out["new_for_you"] = out.apply(why_new, axis=1)
    return out.reset_index(drop=True)
