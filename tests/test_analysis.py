import numpy as np
import pandas as pd
import pytest

from reading_dash.data import load_books, load_candidates, unique_works
from reading_dash.eras import boundary_stability, detect_eras, segment
from reading_dash.insights import blind_spots, headline, pace
from reading_dash.recommend import recommend
from reading_dash.semantic import build_space, galaxy_layout, neighbours


@pytest.fixture(scope="module")
def data():
    df, report = load_books()
    return df, report


def test_every_book_has_metadata(data):
    df, report = data
    assert report.missing_metadata == []
    assert report.unparseable_dates == []
    assert df["id"].is_unique


def test_rereads_detected(data):
    df, _ = data
    rereads = set(df.loc[df["is_reread"], "original_title"])
    assert "Nineteen Eighty-Four" in rereads
    assert "The Subtle Art of Not Giving a F*ck" in rereads


def test_translation_flag(data):
    df, _ = data
    row = df.set_index("id")
    assert not row.loc[3, "translated"]   # Conan Doyle read in English
    assert row.loc[24, "translated"]      # Animal Farm read in Uzbek
    assert row.loc[108, "translated"] is np.False_ or not row.loc[108, "translated"]  # Tolstoy in Russian


def test_segment_recovers_planted_change():
    X = np.vstack([np.tile([1, 0], (10, 1)), np.tile([0, 1], (10, 1))]).astype(float)
    assert segment(X, penalty=1.0, min_size=3) == [10, 20]


def test_segment_single_when_penalty_huge():
    X = np.eye(3)[np.random.default_rng(0).integers(0, 3, 30)]
    assert segment(X, penalty=1e6, min_size=3) == [30]


def test_eras_cover_all_dated_books(data):
    df, _ = data
    eras, assign = detect_eras(df)
    assert eras["books"].sum() == df["dated"].sum()
    assert assign.notna().all()
    assert (eras["start"].diff().dropna() >= pd.Timedelta(0)).all()
    stab = boundary_stability(df)
    assert ((stab >= 0) & (stab <= 1)).all()


def test_insights_run(data):
    df, _ = data
    h = headline(df)
    assert h["reads"] == len(df)
    assert pace(df)["longest_gap_days"] > 0
    assert len(blind_spots(df)) >= 5


def test_space_and_recommendations(data):
    df, _ = data
    works = unique_works(df)
    cand = load_candidates()
    space = build_space(works, cand)
    xy = galaxy_layout(space["read"])
    assert xy.shape == (len(works), 2)
    nb = neighbours(space["read"], 0, k=3)
    assert len(nb) == 3 and all(i != 0 for i, _ in nb)
    for mode in ("similar", "stretch"):
        recs = recommend(works, cand, space["read"], space["cand"], mode=mode, n=5)
        assert len(recs) == 5
        assert not recs["title"].str.lower().isin(works["original_title"].str.lower()).any()


def test_app_runs():
    """Full render of the Streamlit app (skipped when Streamlit isn't installed)."""
    pytest.importorskip("plotly")
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception


def test_load_current(tmp_path):
    from reading_dash.data import load_current

    assert load_current(tmp_path).empty  # no file -> nothing in progress
    (tmp_path / "currently_reading.csv").write_text(
        "title,author,started,read_language,progress\n"
        "A Hero of Our Time,Mikhail Lermontov,01.10.2026,Russian,40\n"
        "Just a title,,,,\n"
        ",,,,\n"
    )
    cur = load_current(tmp_path)
    assert cur["title"].tolist() == ["A Hero of Our Time", "Just a title"]
    assert cur.loc[0, "started"] == pd.Timestamp("2026-10-01")
    assert cur.loc[0, "progress"] == 40
    assert pd.isna(cur.loc[1, "progress"])


def test_add_book_and_current(tmp_path):
    import shutil
    from datetime import date

    from reading_dash.data import DATA_DIR, add_book, load_current, remove_current, save_current

    for name in ["reading_list.csv", "book_metadata.csv"]:
        shutil.copy(DATA_DIR / name, tmp_path / name)
    before, _ = load_books(tmp_path)

    save_current({"title": "Bratya Karamazovy", "author": "Fyodor Dostoevsky", "started": date(2026, 10, 1),
                  "read_language": "Russian", "progress": 20}, tmp_path)
    save_current({"title": "Bratya Karamazovy", "author": "Fyodor Dostoevsky", "started": date(2026, 10, 1),
                  "read_language": "Russian", "progress": 60}, tmp_path)  # update, not a duplicate
    cur = load_current(tmp_path)
    assert len(cur) == 1 and cur.loc[0, "progress"] == 60 and cur.loc[0, "started"] == pd.Timestamp("2026-10-01")

    new_id = add_book({"title": "Bratya Karamazovy", "author": "Fyodor Dostoevsky",
                       "finish_date": date(2026, 11, 20), "read_language": "Russian"},
                      {"original_title": "Bratya Karamazovy", "canonical_author": "Fyodor Dostoevsky",
                       "author_country": "RUS", "author_gender": "M", "original_language": "Russian",
                       "pub_year": 1880, "category": "Literature", "fiction": True, "tags": "Russian classic",
                       "description": "", "verify": False}, tmp_path)
    assert remove_current("bratya karamazovy", tmp_path) and load_current(tmp_path).empty

    after, report = load_books(tmp_path)
    assert len(after) == len(before) + 1 and report.missing_metadata == []
    row = after[after["id"] == new_id].iloc[0]
    assert row["finish_date"] == pd.Timestamp("2026-11-20") and row["country_name"] == "Russia" and not row["translated"]
