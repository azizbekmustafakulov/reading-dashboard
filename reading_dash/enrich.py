"""Fill in book metadata automatically with Claude.

A reader uploads only what they know: title, author and the language they read
it in. This module asks Claude for the rest (original title and language,
author country and gender, first publication year, category, tags and a short
description) in the same shape as data/book_metadata.csv, so the result can be
passed straight to ``data.build_books``.

Needs the ``anthropic`` package and an API key in ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel

from .data import CATEGORIES

MODEL = "claude-opus-5-5"

Category = Literal[tuple(CATEGORIES)]  # type: ignore[valid-type]


class BookMetadata(BaseModel):
    original_title: str
    canonical_author: str
    author_country: str
    author_gender: Literal["M", "F", "Mixed", "Unknown"]
    original_language: str
    pub_year: int | None
    category: Category
    fiction: bool
    tags: list[str]
    description: str
    confidence: Literal["high", "medium", "low"]


# Written by hand (not generated from the pydantic model) because structured
# outputs need additionalProperties: false and every field listed as required.
SCHEMA = {
    "type": "object",
    "properties": {
        "original_title": {"type": "string"},
        "canonical_author": {"type": "string"},
        "author_country": {"type": "string"},
        "author_gender": {"type": "string", "enum": ["M", "F", "Mixed", "Unknown"]},
        "original_language": {"type": "string"},
        "pub_year": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "fiction": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "original_title", "canonical_author", "author_country", "author_gender", "original_language",
        "pub_year", "category", "fiction", "tags", "description", "confidence",
    ],
    "additionalProperties": False,
}

SYSTEM = """You catalogue books for a reading tracker used mostly by readers in Uzbekistan, \
who read in Uzbek, Russian and English. Many books are Uzbek translations of foreign books, \
listed under their Uzbek title.

Given a title, an author as the reader typed it, and the language they read it in, identify the \
work and return its metadata.

Fields:
- original_title: the title in the language the work was first written in. Use Latin script; \
transliterate Russian and Arabic titles (e.g. "Master i Margarita"). For Uzbek, use modern Uzbek \
Latin with a plain apostrophe for o' and g' (e.g. "O'tkan kunlar"). For a translation, give the \
original work's title, not the translated one.
- canonical_author: the author's usual name in Latin script, as English-language references write \
it for foreign authors (e.g. "Fyodor Dostoevsky") and as Uzbek references write it for Uzbek authors. \
Co-authors are joined with " & ".
- author_country: ISO 3166-1 alpha-3 code of the country the author is chiefly associated with \
(first author if several). Empty string if you don't know.
- author_gender: "M" or "F" only if you know this specific author; "Mixed" for co-authors of \
different genders. Never infer gender from the name alone: if you don't know the author, use "Unknown".
- original_language: English name of the language the work was first written in.
- pub_year: year the original work was first published. null if you don't know.
- category: exactly one of: """ + "; ".join(CATEGORIES) + """. Literature covers all fiction and \
poetry. Religion & Ethics covers religious, spiritual and moral works.
- fiction: true for novels, stories, poetry and drama.
- tags: 3 to 6 short English tags about themes and genre.
- description: one plain English sentence (at most 25 words) on what the book is about.
- confidence: "high" if you're sure which work this is, "medium" if fairly sure, "low" if you are \
guessing. Lower it for obscure Uzbek titles you don't recognise rather than inventing details.

Return only facts you are reasonably confident of. A wrong confident answer is worse than \
"Unknown", an empty string or null."""


def _example_block(examples: pd.DataFrame) -> str:
    """Worked examples from an existing, hand-checked catalogue (few-shot)."""
    lines = []
    for r in examples.itertuples():
        answer = {
            "original_title": r.original_title,
            "canonical_author": r.canonical_author,
            "author_country": r.author_country,
            "author_gender": r.author_gender,
            "original_language": r.original_language,
            "pub_year": None if pd.isna(r.pub_year) or r.pub_year == "" else int(float(r.pub_year)),
            "category": r.category,
            "fiction": str(r.fiction).lower() == "true",
            "tags": [t.strip() for t in str(r.tags).split(";") if t.strip()],
            "description": r.description,
            "confidence": "high",
        }
        lines.append(f'Input: title="{r.title}", author="{r.author}", read in {r.read_language}\n'
                     f"Output: {json.dumps(answer, ensure_ascii=False)}")
    return "Examples from the existing catalogue:\n\n" + "\n\n".join(lines)


def enrich_book(client: anthropic.Anthropic, title: str, author: str, read_language: str,
                examples: pd.DataFrame | None = None, effort: str = "medium") -> tuple[BookMetadata | None, str]:
    """Return (metadata, note). metadata is None when Claude declines or the output is unusable."""
    system = [{"type": "text", "text": SYSTEM}]
    if examples is not None and len(examples):
        system.append({"type": "text", "text": _example_block(examples)})
    system[-1]["cache_control"] = {"type": "ephemeral"}  # same prefix for every book in a run

    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if a safety classifier declines, the API retries on a fallback model
        system=system,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f'title="{title}", author="{author}", read in {read_language}'}],
    )
    if response.stop_reason == "refusal":
        return None, "declined"
    if response.stop_reason == "max_tokens":
        return None, "output cut off"
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return BookMetadata.model_validate_json(text), f"model={response.model}"
    except ValueError as e:
        return None, f"invalid output: {e}"


def to_metadata_row(book_id: int, m: BookMetadata) -> dict:
    """One row in the data/book_metadata.csv format. LLM rows are always marked verify=True."""
    return {
        "id": book_id,
        "original_title": m.original_title,
        "canonical_author": m.canonical_author,
        "author_country": m.author_country.upper().strip(),
        "author_gender": m.author_gender,
        "original_language": m.original_language,
        "pub_year": m.pub_year if m.pub_year is not None else "",
        "category": m.category,
        "fiction": m.fiction,
        "tags": ";".join(m.tags),
        "description": m.description,
        "verify": True,
    }
