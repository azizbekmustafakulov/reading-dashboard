"""How good is automatic metadata? Compare Claude's guesses with the hand-made catalogue.

Picks a sample of books from data/reading_list.csv whose metadata was checked by
hand (verify=False), gives Claude only what an uploading user would know
(title, author, read language), and scores each field against
data/book_metadata.csv.

    python scripts/eval_enrichment.py --dry-run     # show the sample and prompt, no API calls
    python scripts/eval_enrichment.py               # run (needs ANTHROPIC_API_KEY)
    python scripts/eval_enrichment.py --n-uzbek 10 --n-translated 8 --n-other 4

Writes reports/enrichment_eval.csv (every field side by side) and
reports/llm_book_metadata.csv (Claude's rows, in book_metadata.csv format).
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reading_dash.data import DATA_DIR, build_books  # noqa: E402

REPORTS = ROOT / "reports"
EXACT = ["author_country", "author_gender", "original_language", "category", "fiction"]
FUZZY = ["original_title", "canonical_author"]


# --------------------------------------------------------------------------- sample
def load_truth() -> pd.DataFrame:
    reads = pd.read_csv(DATA_DIR / "reading_list.csv", dtype=str, keep_default_na=False)
    meta = pd.read_csv(DATA_DIR / "book_metadata.csv", dtype=str, keep_default_na=False)
    df = reads.merge(meta, on="id")
    df = df[df["verify"].str.lower() == "false"]          # only hand-checked rows count as ground truth
    df = df.drop_duplicates("original_title")              # re-reads once
    df["group"] = "other (English/Russian)"
    uz = df["read_language"] == "Uzbek"
    df.loc[uz & (df["original_language"] == "Uzbek"), "group"] = "Uzbek original"
    df.loc[uz & (df["original_language"] != "Uzbek"), "group"] = "Uzbek translation"
    return df.reset_index(drop=True)


def pick(df: pd.DataFrame, n: dict[str, int], seed: int, min_id: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test sample per group (only ids >= min_id), plus few-shot examples by authors not in the sample."""
    pool = df[df["id"].astype(int) >= min_id]
    test = pd.concat([g.sample(min(n[k], len(g)), random_state=seed) for k, g in pool.groupby("group")])
    rest = df[~df["id"].isin(test["id"]) & ~df["canonical_author"].isin(test["canonical_author"])]
    examples = pd.concat([g.sample(1, random_state=seed) for _, g in rest.groupby("group")])
    return test.reset_index(drop=True), examples.reset_index(drop=True)


# --------------------------------------------------------------------------- scoring
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", "", s.lower().replace("&", "and"))
    return re.sub(r"^the ", "", re.sub(r"\s+", " ", s).strip())


def close(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    return a == b or a in b or b in a or SequenceMatcher(None, a, b).ratio() >= 0.85


def score(truth: pd.Series, guess: dict) -> dict:
    out = {f: close(truth[f], guess[f]) for f in FUZZY}
    for f in EXACT:
        out[f] = str(truth[f]).strip().lower() == str(guess[f]).strip().lower()
    t, g = pd.to_numeric(truth["pub_year"], errors="coerce"), pd.to_numeric(guess["pub_year"], errors="coerce")
    out["pub_year"] = bool(pd.notna(t) and pd.notna(g) and t == g)
    out["pub_year_within_2"] = bool(pd.notna(t) and pd.notna(g) and abs(t - g) <= 2)
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-uzbek", type=int, default=6, help="Uzbek-original books in the sample")
    ap.add_argument("--n-translated", type=int, default=5, help="Uzbek translations of foreign books")
    ap.add_argument("--n-other", type=int, default=4, help="books read in English or Russian")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--dry-run", action="store_true", help="show the sample and prompt without calling the API")
    ap.add_argument("--min-id", type=int, default=0, help="sample only books with id >= this")
    ap.add_argument("--answers", type=Path,
                    help="score answers from a JSON file ({id: {field: value}}) instead of calling the API")
    args = ap.parse_args()

    df = load_truth()
    n = {"Uzbek original": args.n_uzbek, "Uzbek translation": args.n_translated,
         "other (English/Russian)": args.n_other}
    test, examples = pick(df, n, args.seed, args.min_id)

    print(f"Ground truth pool: {len(df)} hand-checked books. Sample: {len(test)}. Few-shot examples: {len(examples)}.\n")
    for r in test.itertuples():
        print(f"  [{r.group:<23}] {r.title}  —  {r.author}  ({r.read_language})")
    print("\nExamples shown to the model: " + "; ".join(examples["title"]))

    from reading_dash import enrich  # imported here so --dry-run works without an API key

    if args.dry_run:
        print("\n--- system prompt ---\n" + enrich.SYSTEM + "\n\n" + enrich._example_block(examples))
        return

    import json
    import os

    import anthropic

    answers = json.loads(args.answers.read_text(encoding="utf-8")) if args.answers else None
    if answers is None and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("\nNo API key found. Create one at https://console.anthropic.com, then run:\n"
                 "  export ANTHROPIC_API_KEY=sk-ant-...\n  python scripts/eval_enrichment.py")
    client = None if answers is not None else anthropic.Anthropic()
    rows, meta_rows = [], []
    for i, r in enumerate(test.itertuples(), 1):
        if answers is not None:
            given = answers.get(str(r.id))
            m, note = (enrich.BookMetadata.model_validate(given), "from file") if given else (None, "no answer in file")
        else:
            try:
                m, note = enrich.enrich_book(client, r.title, r.author, r.read_language, examples, effort=args.effort)
            except anthropic.AuthenticationError:
                sys.exit("No valid API key. Set ANTHROPIC_API_KEY and try again.")
            except anthropic.APIError as e:
                m, note = None, f"API error: {e}"
        truth = test.loc[i - 1]
        if m is None:
            print(f"{i:>2}/{len(test)} {r.title[:40]:<40} FAILED ({note})")
            rows.append({"id": r.id, "group": r.group, "title": r.title, "failed": note})
            continue
        guess = enrich.to_metadata_row(int(r.id), m)
        meta_rows.append(guess)
        s = score(truth, guess)
        print(f"{i:>2}/{len(test)} {r.title[:40]:<40} {sum(s[k] for k in EXACT + FUZZY + ['pub_year'])}/8 "
              f"correct, confidence {m.confidence}")
        row = {"id": r.id, "group": r.group, "title": r.title, "author_typed": r.author,
               "confidence": m.confidence, "failed": ""}
        for f in FUZZY + EXACT + ["pub_year"]:
            row[f"{f}_truth"], row[f"{f}_llm"], row[f"{f}_ok"] = truth[f], guess[f], s[f]
        row["pub_year_within_2_ok"] = s["pub_year_within_2"]
        row["tags_truth"], row["tags_llm"] = truth["tags"], guess["tags"]
        row["description_truth"], row["description_llm"] = truth["description"], guess["description"]
        rows.append(row)

    REPORTS.mkdir(exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "enrichment_eval.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(REPORTS / "llm_book_metadata.csv", index=False)

    ok = res[res["failed"] == ""]
    fields = FUZZY + EXACT + ["pub_year", "pub_year_within_2"]
    print(f"\nAccuracy by field ({len(ok)} books scored, {len(res) - len(ok)} failed):")
    table = pd.DataFrame({f: ok.groupby("group")[f"{f}_ok"].mean() for f in fields}).T
    table["all"] = [ok[f"{f}_ok"].mean() for f in fields]
    print((table * 100).round(0).astype("Int64").astype(str).add("%").to_string())
    if "confidence" in ok:
        overall = ok[[f"{f}_ok" for f in FUZZY + EXACT + ["pub_year"]]].mean(axis=1)
        print("\nAverage accuracy by the model's own confidence:")
        print((overall.groupby(ok["confidence"]).agg(["mean", "size"])
               .rename(columns={"mean": "accuracy", "size": "books"})).round(2).to_string())

    # End to end: can the dashboard be built from Claude's metadata alone?
    if meta_rows:
        reads = pd.read_csv(DATA_DIR / "reading_list.csv", dtype={"finish_date": str})
        built, report = build_books(reads[reads["id"].isin(ok["id"].astype(int))], pd.DataFrame(meta_rows).fillna(""))
        print(f"\nDashboard table built from LLM metadata: {len(built)} rows, {len(report.missing_metadata)} missing.")
    print(f"\nDetails: {REPORTS / 'enrichment_eval.csv'}")


if __name__ == "__main__":
    main()
