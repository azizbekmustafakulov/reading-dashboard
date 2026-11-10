"""Loading, merging and cleaning the reading list.

Two hand-maintained files drive everything:

* ``data/reading_list.csv``  - what was read, when, and in which language.
* ``data/book_metadata.csv`` - facts about each book (original language,
  publication year, author country, category, description), keyed by ``id``.

Adding a book means appending one row to each file. A book without a
metadata row still shows up, with "Unknown" fields, and is reported in the
data-quality panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CATEGORIES = [
    "Literature",
    "History & Biography",
    "Economics & Society",
    "Psychology & Behavior",
    "Self-help & Productivity",
    "Business & Entrepreneurship",
    "Religion & Ethics",
]

# One fixed colour per category so every chart reads the same way.
CATEGORY_COLORS = {
    "Literature": "#4C6EF5",
    "History & Biography": "#C2410C",
    "Economics & Society": "#0F9D58",
    "Psychology & Behavior": "#9333EA",
    "Self-help & Productivity": "#E8A317",
    "Business & Entrepreneurship": "#0891B2",
    "Religion & Ethics": "#64748B",
    "Unknown": "#CBD5E1",
}

LANGUAGE_COLORS = {"Uzbek": "#0EA5E9", "English": "#EF4444", "Russian": "#22C55E"}

# ISO-3 code -> (country name, continent). Extend when new countries appear.
COUNTRIES = {
    "UZB": ("Uzbekistan", "Asia"),
    "KGZ": ("Kyrgyzstan", "Asia"),
    "AFG": ("Afghanistan", "Asia"),
    "IRN": ("Iran", "Asia"),
    "PAK": ("Pakistan", "Asia"),
    "IND": ("India", "Asia"),
    "KOR": ("South Korea", "Asia"),
    "ISR": ("Israel", "Asia"),
    "SYR": ("Syria", "Asia"),
    "LBN": ("Lebanon", "Asia"),
    "YEM": ("Yemen", "Asia"),
    "TUR": ("Turkey", "Asia"),
    "RUS": ("Russia", "Europe"),
    "GBR": ("United Kingdom", "Europe"),
    "IRL": ("Ireland", "Europe"),
    "FRA": ("France", "Europe"),
    "ESP": ("Spain", "Europe"),
    "ITA": ("Italy", "Europe"),
    "GRC": ("Greece", "Europe"),
    "AUT": ("Austria", "Europe"),
    "SWE": ("Sweden", "Europe"),
    "CZE": ("Czechia", "Europe"),
    "USA": ("United States", "North America"),
    "CAN": ("Canada", "North America"),
    "BRA": ("Brazil", "South America"),
    "COL": ("Colombia", "South America"),
    "AUS": ("Australia", "Oceania"),
    "TUN": ("Tunisia", "Africa"),
}

# Ancient Greek, Chagatai etc. are all "not the language it was read in".
_LANG_FAMILY = {"Ancient Greek": "Greek"}


@dataclass
class QualityReport:
    undated: list[int] = field(default_factory=list)
    missing_metadata: list[int] = field(default_factory=list)
    needs_verification: list[int] = field(default_factory=list)
    batch_dates: dict[str, int] = field(default_factory=dict)
    out_of_order: int = 0
    unparseable_dates: list[int] = field(default_factory=list)


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_books(data_dir: Path | str = DATA_DIR) -> tuple[pd.DataFrame, QualityReport]:
    """Return one row per reading event, enriched with metadata and derived fields."""
    data_dir = Path(data_dir)
    reads = pd.read_csv(data_dir / "reading_list.csv", dtype={"finish_date": str})
    meta = pd.read_csv(data_dir / "book_metadata.csv", keep_default_na=False)

    report = QualityReport()
    report.missing_metadata = sorted(set(reads["id"]) - set(meta["id"]))

    df = reads.merge(meta, on="id", how="left")

    # Dates are written day-first (dd.mm.yyyy).
    raw = df["finish_date"].fillna("").str.strip()
    df["finish_date"] = pd.to_datetime(raw, format="%d.%m.%Y", errors="coerce")
    report.unparseable_dates = df.loc[(raw != "") & df["finish_date"].isna(), "id"].tolist()
    df["dated"] = df["finish_date"].notna()
    report.undated = df.loc[~df["dated"], "id"].tolist()

    # Fill gaps for books with no metadata row.
    df["original_title"] = df["original_title"].replace("", pd.NA).fillna(df["title"])
    for col, default in [
        ("category", "Unknown"),
        ("author_gender", "Unknown"),
        ("original_language", "Unknown"),
        ("tags", ""),
        ("description", ""),
        ("author_country", ""),
    ]:
        df[col] = df[col].replace("", pd.NA).fillna(default)
    df["canonical_author"] = df["canonical_author"].replace("", pd.NA).fillna(df["author"]).fillna("Unknown")
    df["fiction"] = _to_bool(df["fiction"].fillna("False"))
    df["verify"] = _to_bool(df["verify"].fillna("True"))
    report.needs_verification = df.loc[df["verify"], "id"].tolist()
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce")

    # Geography
    df["country_name"] = df["author_country"].map(lambda c: COUNTRIES.get(c, ("Unknown", "Unknown"))[0])
    df["continent"] = df["author_country"].map(lambda c: COUNTRIES.get(c, ("Unknown", "Unknown"))[1])

    # Read in translation?  (Unknown original language -> not counted either way.)
    orig = df["original_language"].map(lambda x: _LANG_FAMILY.get(x, x))
    df["translated"] = (orig != df["read_language"]) & (orig != "Unknown")

    # Re-reads: the same work (by original title) appearing more than once.
    # Undated rows are the oldest part of the list (read before dates were
    # logged), so they count as the earliest read.
    df = df.sort_values(["finish_date", "id"], na_position="first")
    df["read_count"] = df.groupby("original_title").cumcount() + 1
    df["times_read"] = df.groupby("original_title")["id"].transform("count")
    df["is_reread"] = df["read_count"] > 1
    df = df.sort_values("id").reset_index(drop=True)

    # Derived time fields
    df["year"] = df["finish_date"].dt.year.astype("Int64")
    df["month"] = df["finish_date"].dt.to_period("M")
    df["years_since_pub"] = df["year"].astype("float") - df["pub_year"]
    df["pub_century"] = (df["pub_year"] // 100 + 1).astype("Int64")

    # Data-quality signals
    counts = df.loc[df["dated"], "finish_date"].dt.strftime("%Y-%m-%d").value_counts()
    report.batch_dates = {d: int(n) for d, n in counts.items() if n >= 3}
    dated = df[df["dated"]].sort_values("id")
    report.out_of_order = int((dated["finish_date"].diff().dt.days < 0).sum())

    return df, report


def load_candidates(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    cand = pd.read_csv(Path(data_dir) / "candidates.csv", keep_default_na=False)
    cand["fiction"] = _to_bool(cand["fiction"])
    cand["pub_year"] = pd.to_numeric(cand["pub_year"], errors="coerce")
    cand["country_name"] = cand["author_country"].map(lambda c: COUNTRIES.get(c, ("Unknown", "Unknown"))[0])
    return cand


def unique_works(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct work (re-reads collapsed onto their latest read)."""
    return (
        df.sort_values(["finish_date"], na_position="first")
        .drop_duplicates("original_title", keep="last")
        .sort_values("id")
        .reset_index(drop=True)
    )
