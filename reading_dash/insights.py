"""Headline statistics and automatically written observations."""

from __future__ import annotations

import pandas as pd


def headline(df: pd.DataFrame) -> dict:
    dated = df[df["dated"]]
    works = df.drop_duplicates("original_title")
    return {
        "reads": len(df),
        "works": len(works),
        "dated": len(dated),
        "authors": works["canonical_author"].nunique(),
        "countries": works.loc[works["author_country"] != "", "author_country"].nunique(),
        "read_languages": df["read_language"].nunique(),
        "original_languages": works.loc[works["original_language"] != "Unknown", "original_language"].nunique(),
        "rereads": int(df["is_reread"].sum()),
        "translated_share": float(df["translated"].mean()),
        "fiction_share": float(works["fiction"].mean()),
        "first": dated["finish_date"].min(),
        "last": dated["finish_date"].max(),
    }


def pace(df: pd.DataFrame) -> dict:
    dated = df[df["dated"]].sort_values("finish_date")
    if dated.empty:
        return {}
    months = dated.groupby("month").size()
    all_months = pd.period_range(months.index.min(), months.index.max(), freq="M")
    full = months.reindex(all_months, fill_value=0)

    # Longest run of consecutive months with at least one finished book.
    best_len, best_end, run = 0, None, 0
    for m, c in full.items():
        run = run + 1 if c > 0 else 0
        if run > best_len:
            best_len, best_end = run, m

    gaps = dated["finish_date"].diff().dt.days.reset_index(drop=True)
    has_gap = gaps.notna().any()
    gpos = int(gaps.idxmax()) if has_gap else 0
    return {
        "busiest_month": str(months.idxmax()),
        "busiest_count": int(months.max()),
        "active_months": int((full > 0).sum()),
        "total_months": len(full),
        "per_active_month": float(months.mean()),
        "streak_months": best_len,
        "streak_end": str(best_end) if best_end is not None else "",
        "longest_gap_days": int(gaps.max()) if has_gap else 0,
        "gap_from": dated["finish_date"].iloc[gpos - 1] if has_gap else None,
        "gap_to": dated["finish_date"].iloc[gpos] if has_gap else None,
        "monthly": full,
    }


def blind_spots(df: pd.DataFrame) -> list[dict]:
    """Observations about what the reading list under-represents."""
    works = df.drop_duplicates("original_title")
    known_gender = works[works["author_gender"].isin(["M", "F", "Mixed"])]
    women = known_gender["author_gender"].isin(["F", "Mixed"]).mean() if len(known_gender) else 0
    women_titles = works.loc[works["author_gender"].isin(["F", "Mixed"]), "canonical_author"].unique()

    continents = works["continent"].value_counts()
    missing_cont = [c for c in ["Africa", "South America", "Oceania", "Asia", "Europe", "North America"] if c not in continents]
    asia_ex_ca = works[(works["continent"] == "Asia") & (~works["author_country"].isin(["UZB"]))]

    lit = works[works["category"] == "Literature"]
    lit_countries = lit["country_name"].value_counts()

    orig = works.loc[works["original_language"] != "Unknown", "original_language"].value_counts()
    centuries = works["pub_century"].dropna().astype(int)
    pre_1900 = (works["pub_year"] < 1900).mean()

    notes = [
        {
            "title": "Women authors (incl. co-authors)",
            "value": f"{women:.0%}",
            "detail": f"{len(women_titles)} of {works['canonical_author'].nunique()} authors: " + ", ".join(women_titles) + ".",
            "severity": "high" if women < 0.2 else "medium" if women < 0.35 else "low",
        },
        {
            "title": "Korean literature",
            "value": str(int((lit["author_country"] == "KOR").sum())),
            "detail": "No Korean novels yet, despite living in Korea. Korea's only appearance is W. Chan Kim's business strategy book."
            if (lit["author_country"] == "KOR").sum() == 0
            else "Korean fiction is on the list.",
            "severity": "high" if (lit["author_country"] == "KOR").sum() == 0 else "low",
        },
        {
            "title": "Where the fiction comes from",
            "value": f"{lit_countries.size} countries",
            "detail": "Top: " + ", ".join(f"{c} ({n})" for c, n in lit_countries.head(4).items()) + ".",
            "severity": "medium",
        },
        {
            "title": "Missing continents",
            "value": str(len(missing_cont)),
            "detail": ("No authors from " + ", ".join(missing_cont) + ".") if missing_cont else "Every continent is represented.",
            "severity": "medium" if missing_cont else "low",
        },
        {
            "title": "Original languages",
            "value": f"{orig.size}",
            "detail": "Originally written in: " + ", ".join(f"{l} ({n})" for l, n in orig.items() if l != "Unknown") + ".",
            "severity": "low",
        },
        {
            "title": "Older than 1900",
            "value": f"{pre_1900:.0%}",
            "detail": f"Publication centuries span {centuries.min()}–{centuries.max()} (from Marcus Aurelius to 2023).",
            "severity": "low",
        },
        {
            "title": "Asia beyond Uzbekistan",
            "value": str(len(asia_ex_ca)),
            "detail": "By country: "
            + ", ".join(f"{c} ({n})" for c, n in asia_ex_ca["country_name"].value_counts().items())
            + ". East and South Asian writing is almost absent.",
            "severity": "medium",
        },
    ]
    return notes


def era_story(eras: pd.DataFrame) -> str:
    if eras.empty:
        return ""
    parts = [f"**{r.label}** ({r.start:%b %Y} – {r.end:%b %Y}, {r.books} books)" for r in eras.itertuples()]
    return " → ".join(parts)
