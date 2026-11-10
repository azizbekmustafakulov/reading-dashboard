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

import csv
from dataclasses import dataclass, field
from datetime import date
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

# One fixed, muted colour per category so every chart reads the same way.
CATEGORY_COLORS = {
    "Literature": "#4E79A7",
    "History & Biography": "#D4846A",
    "Economics & Society": "#59A08B",
    "Psychology & Behavior": "#9C7FB8",
    "Self-help & Productivity": "#E3B04B",
    "Business & Entrepreneurship": "#6FB1C7",
    "Religion & Ethics": "#9A928A",
    "Unknown": "#D6D3D1",
}

# Languages are a different dimension, so they get shades of one hue instead of new colours.
LANGUAGE_COLORS = {"Uzbek": "#2F4B7C", "English": "#7A93B8", "Russian": "#C3CFDF"}

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
    return build_books(reads, meta)


def build_books(reads: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Build the analysis table from any reader's list.

    ``reads`` has the reading_list.csv columns (id, title, author, finish_date as
    dd.mm.yyyy text, read_language); ``meta`` has the book_metadata.csv columns,
    keyed by the same id. Nothing here depends on whose list it is.
    """
    reads = reads.copy()
    reads["finish_date"] = reads["finish_date"].astype("string")
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


def load_current(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Books in progress, from ``data/currently_reading.csv``.

    Columns: ``title,author,started (dd.mm.yyyy),read_language,progress (0-100)``.
    Only title is required. A missing file means nothing is in progress.
    These books are kept out of every statistic until they move to reading_list.csv.
    """
    cols = ["title", "author", "started", "read_language", "progress"]
    path = Path(data_dir) / "currently_reading.csv"
    if not path.exists():
        return pd.DataFrame(columns=cols)
    cur = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in cols:
        if col not in cur:
            cur[col] = ""
        cur[col] = cur[col].str.strip()
    cur = cur[cur["title"] != ""].reset_index(drop=True)
    cur["started"] = pd.to_datetime(cur["started"], format="%d.%m.%Y", errors="coerce")
    cur["progress"] = pd.to_numeric(cur["progress"], errors="coerce").clip(0, 100)
    return cur[cols]


def unique_works(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct work (re-reads collapsed onto their latest read)."""
    return (
        df.sort_values(["finish_date"], na_position="first")
        .drop_duplicates("original_title", keep="last")
        .sort_values("id")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- writing
READ_COLS = ["id", "title", "author", "finish_date", "read_language"]
META_COLS = ["id", "original_title", "canonical_author", "author_country", "author_gender", "original_language",
             "pub_year", "category", "fiction", "tags", "description", "verify"]
CURRENT_COLS = ["title", "author", "started", "read_language", "progress"]


def _fmt_date(value) -> str:
    return value.strftime("%d.%m.%Y") if isinstance(value, date) else (value or "")


def _append_row(path: Path, default_cols: list[str], row: dict) -> None:
    """Append one row, keeping the file's own column order. Creates the file if needed."""
    has_content = path.exists() and path.stat().st_size > 0
    cols = default_cols
    if has_content:
        with open(path, encoding="utf-8", newline="") as f:
            cols = next(csv.reader(f))
        with open(path, "rb") as f:
            f.seek(-1, 2)
            missing_newline = f.read(1) != b"\n"
    with open(path, "a", encoding="utf-8", newline="") as f:
        if has_content and missing_newline:
            f.write("\n")
        writer = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        if not has_content:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in cols})


def add_book(read: dict, meta: dict | None, data_dir: Path | str = DATA_DIR) -> int:
    """Add a finished book to reading_list.csv (and its metadata row, if given). Returns the new id.

    ``read`` needs title, author, read_language and optionally finish_date (a date or dd.mm.yyyy text).
    """
    data_dir = Path(data_dir)
    ids = pd.read_csv(data_dir / "reading_list.csv", usecols=["id"])["id"]
    new_id = int(ids.max()) + 1 if len(ids) else 1
    _append_row(data_dir / "reading_list.csv", READ_COLS,
                {**read, "id": new_id, "finish_date": _fmt_date(read.get("finish_date"))})
    if meta is not None:
        _append_row(data_dir / "book_metadata.csv", META_COLS, {**meta, "id": new_id})
    return new_id


def save_current(row: dict, data_dir: Path | str = DATA_DIR) -> None:
    """Add a book to currently_reading.csv, or update it if the title is already there."""
    path = Path(data_dir) / "currently_reading.csv"
    remove_current(row["title"], data_dir)
    _append_row(path, CURRENT_COLS, {**row, "started": _fmt_date(row.get("started"))})


def remove_current(title: str, data_dir: Path | str = DATA_DIR) -> bool:
    """Drop a title (case-insensitive) from currently_reading.csv. Returns True if one was removed."""
    path = Path(data_dir) / "currently_reading.csv"
    if not path.exists() or path.stat().st_size == 0:
        return False
    cur = pd.read_csv(path, dtype=str, keep_default_na=False)
    keep = cur["title"].str.strip().str.lower() != title.strip().lower()
    if keep.all():
        return False
    cur[keep].to_csv(path, index=False, lineterminator="\n")
    return True
