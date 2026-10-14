"""BM25 lexical index for session memory chunks.

Provides sub-millisecond keyword search over chunk text.
Zero VRAM, pure CPU, designed for the 16GB constrained
environment.

BM25 (Best Matching 25) is the standard lexical ranking
function for information retrieval.  It scores documents
by term frequency normalized by inverse document frequency,
with length normalization.  This implementation uses the
classic Okapi BM25 variant with configurable k1 and b.

Fields and weights:
* ``text``         — weight 1.0 (primary search target)
* ``prev_context`` — weight 0.3 (continuity, shouldn't dominate)
* ``next_context`` — weight 0.3 (continuity, shouldn't dominate)
* ``symbol``       — weight 2.0 (code self-reference via names)

The weighted field scores are combined linearly:
``score = w_text * bm25(text) + w_prev * bm25(prev) + ...``

Stopwords are removed using a small hardcoded list (no
external dependency).  Tokenization is simple regex-based
word splitting, which works well for code and prose alike.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .chunker import Chunk


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BM25 parameters (Okapi defaults)
K1 = 1.5       # term frequency saturation
B = 0.75       # length normalization

# Field weights
WEIGHT_TEXT = 1.0
WEIGHT_PREV = 0.3
WEIGHT_NEXT = 0.3
WEIGHT_SYMBOL = 2.0

# Stopwords (small set, no external dep)
_STOPWORDS = frozenset({
    # English articles / prepositions / conjunctions
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "up", "down",
    "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "or", "but", "if", "else", "when", "where",
    "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "also",
    # Code keywords that are too common to be discriminative
    "def", "return", "self", "this", "true", "false", "none",
    "true", "null", "void", "int", "str", "len",
})

# Tokenizer: words, identifiers, numbers
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
# CamelCase splitter: splits getCellFromMouse into get, cell, from, mouse
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens.

    Splits on non-alphanumeric boundaries, lowercases,
    and removes stopwords.  Works for both code and prose.
    """
    if not text:
        return []
    raw = _TOKEN_RE.findall(text)
    # Split camelCase identifiers into sub-words for better matching.
    # getCellFromMouse → get, cell, from, mouse
    # XMLHttpRequest → xml, http, request
    # hello_world stays as hello_world (snake_case is not split)
    expanded: list[str] = []
    for token in raw:
        # Only split if there are no underscores (snake_case)
        # and the token has mixed case or starts with uppercase
        if "_" not in token:
            sub_words = _CAMEL_RE.findall(token)
            if len(sub_words) > 1:
                expanded.extend(sub_words)
            else:
                expanded.append(token)
        else:
            expanded.append(token)
    return [t.lower() for t in expanded if t.lower() not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class BM25Index:
    """In-memory BM25 index over Chunk objects.

    The index is built incrementally as chunks are inserted.
    Search returns the top-K chunk_ids ranked by BM25 score.

    The index stores:
    * Per-field token lists for each chunk
    * Per-field document frequencies for IDF computation
    * Per-field average document length for length normalization
    * Chunk ID → Chunk mapping for retrieval

    The index is designed for ~1000 chunks (a long session).
    Memory usage is proportional to total chunk text, which
    at 500 tokens/chunk * 1000 chunks ≈ 500K tokens ≈ 2MB.
    """

    def __init__(self) -> None:
        # chunk_id → Chunk
        self._chunks: dict[str, Chunk] = {}

        # Per-field token storage: chunk_id → field → list[str]
        self._field_tokens: dict[str, dict[str, list[str]]] = {}

        # Per-field document frequency: token → doc count
        self._df: dict[str, Counter] = {}

        # Per-field total document length (for avgdl)
        self._field_lengths: dict[str, list[int]] = {}

        # Number of documents
        self._n_docs: int = 0

        # Cached average document length per field
        self._avgdl: dict[str, float] = {}

        # Dirty flag for avgdl cache
        self._dirty: bool = False

    def insert(self, chunk: Chunk) -> None:
        """Insert a single chunk into the index."""
        if chunk.chunk_id in self._chunks:
            return  # already indexed

        self._chunks[chunk.chunk_id] = chunk

        fields = {
            "text": chunk.text,
            "prev_context": chunk.prev_context,
            "next_context": chunk.next_context,
            "symbol": chunk.symbol or "",
        }

        field_tokens: dict[str, list[str]] = {}
        for field_name, field_text in fields.items():
            tokens = tokenize(field_text)
            field_tokens[field_name] = tokens

            # Update document frequency
            if field_name not in self._df:
                self._df[field_name] = Counter()
            unique = set(tokens)
            for t in unique:
                self._df[field_name][t] += 1

            # Track field length
            if field_name not in self._field_lengths:
                self._field_lengths[field_name] = []
            self._field_lengths[field_name].append(len(tokens))

        self._field_tokens[chunk.chunk_id] = field_tokens
        self._n_docs += 1
        self._dirty = True

    def insert_many(self, chunks: list[Chunk]) -> None:
        """Insert multiple chunks."""
        for chunk in chunks:
            self.insert(chunk)

    def remove(self, chunk_id: str) -> None:
        """Remove a chunk from the index."""
        if chunk_id not in self._chunks:
            return

        # Remove from field tokens
        ft = self._field_tokens.pop(chunk_id, {})
        for field_name, tokens in ft.items():
            unique = set(tokens)
            for t in unique:
                if field_name in self._df:
                    self._df[field_name][t] -= 1
                    if self._df[field_name][t] <= 0:
                        del self._df[field_name][t]
            if field_name in self._field_lengths and self._field_lengths[field_name]:
                self._field_lengths[field_name].pop()

        self._chunks.pop(chunk_id, None)
        self._n_docs -= 1
        self._dirty = True

    def remove_turn(self, turn_id: str) -> None:
        """Remove all chunks belonging to a turn."""
        to_remove = [
            cid for cid, chunk in self._chunks.items()
            if chunk.turn_id == turn_id
        ]
        for cid in to_remove:
            self.remove(cid)

    def _ensure_avgdl(self) -> None:
        """Recompute cached average document lengths if dirty."""
        if not self._dirty:
            return
        for field_name, lengths in self._field_lengths.items():
            self._avgdl[field_name] = sum(lengths) / max(len(lengths), 1)
        self._dirty = False

    def _bm25_field_score(
        self,
        query_tokens: list[str],
        field_name: str,
        k1: float = K1,
        b: float = B,
    ) -> dict[str, float]:
        """Compute BM25 scores for a query on a single field.

        Returns a dict of chunk_id → score for this field.
        """
        self._ensure_avgdl()
        n_docs = self._n_docs
        if n_docs == 0:
            return {}

        avgdl = self._avgdl.get(field_name, 0.0)
        df = self._df.get(field_name, Counter())

        scores: dict[str, float] = {}

        for chunk_id, ft in self._field_tokens.items():
            tokens = ft.get(field_name, [])
            if not tokens:
                continue

            dl = len(tokens)
            tf_counter = Counter(tokens)

            score = 0.0
            for qt in query_tokens:
                tf = tf_counter.get(qt, 0)
                if tf == 0:
                    continue
                # IDF: Okapi BM25 variant
                n_qi = df.get(qt, 0)
                idf = math.log(1 + (n_docs - n_qi + 0.5) / (n_qi + 0.5))
                # TF saturation
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1)))
                score += idf * tf_norm

            if score > 0:
                scores[chunk_id] = score

        return scores

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        min_score: float = 0.01,
    ) -> list[tuple[Chunk, float]]:
        """Search the index for the top-K chunks matching the query.

        Parameters
        ----------
        query
            The search query string.
        k
            Maximum number of results to return.
        min_score
            Minimum BM25 score threshold.  Results below
            this are filtered out.

        Returns
        -------
        list[tuple[Chunk, float]]
            List of (chunk, score) pairs, highest score first.
        """
        if not query or not query.strip() or self._n_docs == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Score each field separately, then combine with weights
        field_scores: dict[str, dict[str, float]] = {}
        weights = {
            "text": WEIGHT_TEXT,
            "prev_context": WEIGHT_PREV,
            "next_context": WEIGHT_NEXT,
            "symbol": WEIGHT_SYMBOL,
        }

        for field_name, weight in weights.items():
            fs = self._bm25_field_score(query_tokens, field_name)
            for cid, score in fs.items():
                if cid not in field_scores:
                    field_scores[cid] = {}
                field_scores[cid][field_name] = score * weight

        # Combine field scores per chunk
        combined: dict[str, float] = {}
        for cid, scores in field_scores.items():
            combined[cid] = sum(scores.values())

        # Filter by min_score and sort
        results = [
            (cid, score)
            for cid, score in combined.items()
            if score >= min_score
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-K as (Chunk, score) pairs
        return [
            (self._chunks[cid], score)
            for cid, score in results[:k]
        ]

    def search_by_turn(
        self,
        query: str,
        turn_id: str,
        *,
        k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Search within chunks belonging to a specific turn."""
        all_results = self.search(query, k=k * 3)
        return [
            (chunk, score)
            for chunk, score in all_results
            if chunk.turn_id == turn_id
        ][:k]

    def get_chunks_for_turn(self, turn_id: str) -> list[Chunk]:
        """Return all chunks for a given turn, ordered by seq."""
        return [
            self._chunks[cid]
            for cid in sorted(
                c.chunk_id for c in self._chunks.values()
                if c.turn_id == turn_id
            )
        ]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Get a single chunk by ID."""
        return self._chunks.get(chunk_id)

    def clear(self) -> None:
        """Remove all chunks from the index."""
        self._chunks.clear()
        self._field_tokens.clear()
        self._df.clear()
        self._field_lengths.clear()
        self._avgdl.clear()
        self._n_docs = 0
        self._dirty = False

    def __len__(self) -> int:
        return self._n_docs

    @property
    def total_tokens(self) -> int:
        """Estimated total tokens across all indexed chunks."""
        return sum(c.token_estimate for c in self._chunks.values())

    def stats(self) -> dict[str, Any]:
        """Return index statistics for debugging."""
        self._ensure_avgdl()
        return {
            "n_chunks": self._n_docs,
            "n_turns": len(set(c.turn_id for c in self._chunks.values())),
            "total_tokens": self.total_tokens,
            "avgdl": dict(self._avgdl),
            "vocab_size": {
                field: len(df) for field, df in self._df.items()
            },
        }