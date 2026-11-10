"""A shared text space for books: the galaxy map, neighbours and recommendations.

Each book is represented by its tags (weighted double), category and
description, vectorised with TF-IDF. Read books and recommendation
candidates are fitted together so that they live in the same space.

TF-IDF is used because it needs no model download and works offline. To
use sentence embeddings instead, swap ``vectorize`` for a
sentence-transformers encoder; everything downstream takes a matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def book_text(row: pd.Series) -> str:
    tags = [t.strip() for t in str(row.get("tags", "")).split(";") if t.strip()]
    tag_tokens = " ".join(t.replace(" ", "_").replace("-", "_") for t in tags)
    tag_words = " ".join(tags)
    return " ".join([tag_tokens, tag_tokens, tag_words, str(row.get("category", "")), str(row.get("description", ""))])


def vectorize(texts: list[str]):
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2), min_df=1)
    return vec.fit_transform(texts), vec


def build_space(works: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    texts = [book_text(r) for _, r in works.iterrows()] + [book_text(r) for _, r in candidates.iterrows()]
    M, vec = vectorize(texts)
    n = len(works)
    return {"matrix": M, "read": M[:n], "cand": M[n:], "vectorizer": vec}


def galaxy_layout(matrix, seed: int = 42) -> np.ndarray:
    """2-D layout: SVD to a dense space, then t-SNE."""
    n = matrix.shape[0]
    k = max(2, min(30, n - 1, matrix.shape[1] - 1))
    dense = normalize(TruncatedSVD(n_components=k, random_state=seed).fit_transform(matrix))
    perplexity = max(5, min(15, (n - 1) // 4))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed, metric="cosine").fit_transform(dense)


def neighbours(matrix, idx: int, k: int = 5) -> list[tuple[int, float]]:
    sims = cosine_similarity(matrix[idx], matrix).ravel()
    sims[idx] = -1
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]
