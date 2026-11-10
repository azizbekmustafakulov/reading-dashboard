"""Offline tests for the enrichment pipeline (no API calls)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("anthropic")

from reading_dash.data import build_books, load_books
from reading_dash.enrich import BookMetadata, to_metadata_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eval_enrichment import close, score  # noqa: E402


def test_title_matching_tolerates_spelling():
    assert close("Ulug‘bek xazinasi", "Ulug'bek xazinasi")
    assert close("The 7 Habits of Highly Effective People", "7 Habits of Highly Effective People")
    assert close("Chan Kim & Renée Mauborgne", "Chan Kim and Renee Mauborgne")
    assert not close("Animal Farm", "Nineteen Eighty-Four")
    assert not close("", "Anything")


def test_dashboard_builds_from_llm_metadata_alone():
    """A new reader with no hand-made catalogue still gets a working table."""
    reads = pd.DataFrame({"id": [1, 2], "title": ["Kecha va kunduz", "Unknown book"],
                          "author": ["Cho‘lpon", "Someone"], "finish_date": ["01.02.2024", ""],
                          "read_language": ["Uzbek", "Uzbek"]})
    m = BookMetadata(original_title="Kecha va kunduz", canonical_author="Cho'lpon", author_country="uzb",
                     author_gender="M", original_language="Uzbek", pub_year=1936, category="Literature",
                     fiction=True, tags=["Uzbek classic", "novel"], description="A novel.", confidence="high")
    row = to_metadata_row(1, m)
    assert row["author_country"] == "UZB" and row["verify"] is True and row["tags"] == "Uzbek classic;novel"
    df, report = build_books(reads, pd.DataFrame([row]))
    assert len(df) == 2 and report.missing_metadata == [2]
    assert df.loc[df["id"] == 1, "country_name"].item() == "Uzbekistan"
    assert df.loc[df["id"] == 2, "category"].item() == "Unknown"


def test_score_against_own_catalogue():
    df, _ = load_books()
    truth = df[df["id"] == 46].iloc[0].astype(str)  # O'tkan kunlar
    guess = {**truth.to_dict(), "pub_year": "1925"}
    s = score(truth, guess)
    assert s["original_title"] and s["category"] and not s["pub_year"] and s["pub_year_within_2"]
