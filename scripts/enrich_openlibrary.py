"""Suggest metadata for books from the Open Library API.

For each book in data/reading_list.csv, look it up on Open Library and write
what it finds (first publication year, page count, subjects) to
data/enrichment_suggestions.csv, next to the values already in
data/book_metadata.csv. Nothing is overwritten: review the suggestions and
copy across the ones you trust.

    python scripts/enrich_openlibrary.py            # all books
    python scripts/enrich_openlibrary.py --ids 108 109

Uses only the standard library. Uzbek-language titles are searched by their
original title from book_metadata.csv, which matches far more often.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
API = "https://openlibrary.org/search.json"
HEADERS = {"User-Agent": "reading-atlas/1.0 (personal reading dashboard)"}


def search(title: str, author: str) -> dict | None:
    params = {"title": title, "limit": 1, "fields": "title,author_name,first_publish_year,number_of_pages_median,subject,key"}
    if author and author != "Unknown":
        params["author"] = author.split("&")[0].split(",")[0].strip()
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as r:
        docs = json.load(r).get("docs", [])
    return docs[0] if docs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int)
    args = ap.parse_args()

    meta = {int(r["id"]): r for r in csv.DictReader(open(DATA / "book_metadata.csv", encoding="utf-8"))}
    reads = list(csv.DictReader(open(DATA / "reading_list.csv", encoding="utf-8")))
    out = []
    for r in reads:
        i = int(r["id"])
        if args.ids and i not in args.ids:
            continue
        m = meta.get(i, {})
        title = m.get("original_title") or r["title"]
        author = m.get("canonical_author") or r["author"]
        try:
            doc = search(title, author)
        except Exception as e:  # network errors etc.
            print(f"{i:>3} {title[:40]:<40} error: {e}")
            continue
        row = {
            "id": i,
            "title": title,
            "current_pub_year": m.get("pub_year", ""),
            "ol_title": doc.get("title", "") if doc else "",
            "ol_author": ", ".join(doc.get("author_name", [])[:2]) if doc else "",
            "ol_first_publish_year": doc.get("first_publish_year", "") if doc else "",
            "ol_pages_median": doc.get("number_of_pages_median", "") if doc else "",
            "ol_subjects": "; ".join(doc.get("subject", [])[:8]) if doc else "",
            "ol_url": f"https://openlibrary.org{doc['key']}" if doc and doc.get("key") else "",
        }
        out.append(row)
        print(f"{i:>3} {title[:40]:<40} -> {row['ol_title'][:40] or 'no match'} ({row['ol_first_publish_year']})")
        time.sleep(0.5)  # be polite to the API

    if out:
        with open(DATA / "enrichment_suggestions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"\nWrote {len(out)} suggestions to data/enrichment_suggestions.csv")


if __name__ == "__main__":
    main()
