"""Turn summary (Layer 2) and heuristic memory relevance gate.

Two distinct concerns live in this module:

1. **TurnSummary** — a compact per-turn record built
   deterministically from the user prompt and chunk
   metadata.  No model call required.  Used by the
   memory injection assembler to decide which past
   turns are relevant to the current prompt.

2. **heuristic_memory_relevance** — a cheap, deterministic
   function that examines the current user prompt and
   recent turn summaries to decide whether memory
   injection is needed at all, and if so, how broad it
   should be.  Mirrors the existing routing heuristic
   pattern: cheap signals first, model only when signals
   are insufficient.

The TurnSummary is designed to be produced by either:
* ``deterministic_extract()`` — instant, slightly noisy
* Router model piggyback — when the router already ran
  for routing, the summary is a few extra JSON fields
* NPU concurrent ingestion — when an NPU is available,
  the router ingests the stream and produces the summary
  concurrently (invisible to the user)

All three paths produce the same ``TurnSummary`` dataclass,
so the downstream code is agnostic to the source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .chunker import Chunk
from .session_memory import estimate_tokens


# ---------------------------------------------------------------------------
# TurnSummary data structure
# ---------------------------------------------------------------------------

@dataclass
class TurnSummary:
    """Compact per-turn record for Layer 2 indexing.

    Built at freeze time from the user prompt and the
    model output's chunk metadata.  Used by the memory
    injection assembler to find relevant past turns.

    Attributes
    ----------
    turn_id
        Foreign key to the parent turn.
    seq
        Chronological position (turn number).
    intent
        One-line description of what the turn was about.
        Extracted from the user prompt (first sentence or
        first non-empty line after stripping code).
    symbols
        Function/class names extracted from code chunks.
    keywords
        Top-N topical keywords (TF, stopwords removed).
    headline
        First code line or first prose sentence from the
        model output.  Helps the user/model identify the turn.
    domain
        Content domain: ``"python"``, ``"web"``, ``"compiled"``,
        ``"prose"``, ``"mixed"``.
    task_type
        ``"new_code"``, ``"bug_fix"``, ``"review"``, ``"explanation"``,
        ``"other"``.
    timestamp
        When the turn was frozen.
    """

    turn_id: str
    seq: int
    intent: str = ""
    symbols: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    headline: str = ""
    domain: str = "prose"
    task_type: str = "other"
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "seq": self.seq,
            "intent": self.intent,
            "symbols": self.symbols,
            "keywords": self.keywords,
            "headline": self.headline,
            "domain": self.domain,
            "task_type": self.task_type,
            "timestamp": self.timestamp,
        }

    @property
    def search_text(self) -> str:
        """Concatenated text for BM25 indexing."""
        parts = [self.intent, " ".join(self.symbols), self.headline, " ".join(self.keywords)]
        return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Stopwords for keyword extraction
# ---------------------------------------------------------------------------

_KEYWORD_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "can", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "and", "or", "but", "if",
    "else", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "own", "same", "so", "than", "too", "very", "just",
    "also", "this", "that", "these", "those", "it", "its", "i",
    "you", "we", "they", "he", "she", "me", "him", "her", "us",
    "them", "my", "your", "our", "their",
    # Code keywords too common to be discriminative
    "def", "return", "self", "class", "import", "from", "true",
    "false", "none", "null", "void", "int", "str", "len", "print",
    "elif", "pass", "with", "try", "except", "raise", "while",
    "for", "if", "else", "in", "not", "and", "or", "is",
})

# Identifier extraction
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# Code symbol definitions
_DEF_RE = re.compile(
    r"\b(?:def|class|function|fn|func|pub\s+fn|struct|enum|interface)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
# Sentence boundary
_SENTENCE_RE = re.compile(r"[.!?]\s+")


# ---------------------------------------------------------------------------
# Deterministic extractor
# ---------------------------------------------------------------------------

def deterministic_extract(
    turn_id: str,
    seq: int,
    user_prompt: str,
    model_output: str,
    chunks: list[Chunk],
    routing_decision: dict[str, Any] | None = None,
    timestamp: float = 0.0,
) -> TurnSummary:
    """Build a TurnSummary deterministically from turn data.

    This is the zero-cost fallback.  No model call required.
    The extraction is slightly noisy but always available.

    Parameters
    ----------
    turn_id
        The turn's unique ID.
    seq
        The turn's chronological position.
    user_prompt
        The original user prompt (unmodified).
    model_output
        The full model output.
    chunks
        The chunks extracted from model_output.
    routing_decision
        Optional routing metadata (domain, task_type, etc.).
    timestamp
        When the turn was frozen.
    """
    # Intent: first sentence of user prompt (after stripping code fences)
    intent = _extract_intent(user_prompt)

    # Symbols: from code chunks + regex on model output
    symbols: list[str] = []
    for chunk in chunks:
        if chunk.symbol:
            symbols.append(chunk.symbol)
    # Also regex-scan the model output for definitions we might have missed
    for m in _DEF_RE.finditer(model_output):
        name = m.group(1)
        if name not in symbols:
            symbols.append(name)
    # Deduplicate, limit to 20
    seen = set()
    unique_symbols: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)
        if len(unique_symbols) >= 20:
            break
    symbols = unique_symbols

    # Keywords: TF over model output + user prompt, stopwords removed
    keywords = _extract_keywords(model_output + " " + user_prompt, max_k=15)

    # Headline: first code line or first prose sentence
    headline = _extract_headline(model_output, chunks)

    # Domain and task_type from routing decision or inference
    domain = "prose"
    task_type = "other"
    if routing_decision:
        domain = routing_decision.get("domain", domain)
        task_type = routing_decision.get("task_type", task_type)
    # Infer from chunks if routing didn't provide it
    if domain == "prose" and chunks:
        code_chunks = [c for c in chunks if c.kind == "code"]
        if code_chunks:
            domain = code_chunks[0].language or "code"
    if task_type == "other":
        diff_chunks = [c for c in chunks if c.kind == "diff_hunk"]
        if diff_chunks:
            task_type = "bug_fix"
        elif any(c.kind == "code" for c in chunks):
            task_type = "new_code"
        elif any(c.kind == "prose" for c in chunks):
            task_type = "explanation"

    return TurnSummary(
        turn_id=turn_id,
        seq=seq,
        intent=intent,
        symbols=symbols,
        keywords=keywords,
        headline=headline,
        domain=domain,
        task_type=task_type,
        timestamp=timestamp,
    )


def _extract_intent(user_prompt: str) -> str:
    """Extract a one-line intent from the user prompt.

    Strips code fences, takes the first sentence.
    """
    # Strip code fences
    prompt_clean = re.sub(r"```[^\n]*\n.*?```", "", user_prompt, flags=re.DOTALL)
    prompt_clean = prompt_clean.strip()
    if not prompt_clean:
        # If prompt was entirely code, use a placeholder
        return "[code-only prompt]"

    # Take the first sentence
    sentences = _SENTENCE_RE.split(prompt_clean)
    first = sentences[0].strip() if sentences else prompt_clean
    # Truncate to ~80 chars
    if len(first) > 80:
        first = first[:77] + "..."
    return first


def _extract_keywords(text: str, max_k: int = 15) -> list[str]:
    """Extract top keywords by term frequency."""
    identifiers = _IDENTIFIER_RE.findall(text)
    # Filter stopwords, lowercase
    filtered = [w for w in identifiers if w.lower() not in _KEYWORD_STOPWORDS]
    # Count
    counts: dict[str, int] = {}
    for w in filtered:
        wl = w.lower()
        counts[wl] = counts.get(wl, 0) + 1
    # Sort by frequency, then alphabetically
    sorted_kw = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [kw for kw, _ in sorted_kw[:max_k]]


def _extract_headline(model_output: str, chunks: list[Chunk]) -> str:
    """Extract a headline: first code line or first prose sentence."""
    # Try code chunks first
    for chunk in chunks:
        if chunk.kind == "code" and chunk.text.strip():
            first_line = chunk.text.strip().splitlines()[0]
            return first_line[:120]
    # Fall back to prose chunks
    for chunk in chunks:
        if chunk.kind == "prose" and chunk.text.strip():
            first_line = chunk.text.strip().splitlines()[0]
            return first_line[:120]
    # Fall back to first non-empty line of model output
    for line in model_output.splitlines():
        if line.strip():
            return line.strip()[:120]
    return ""


# ---------------------------------------------------------------------------
# Turn summary index (Layer 2)
# ---------------------------------------------------------------------------

class TurnSummaryIndex:
    """BM25 index over TurnSummary records.

    This is the Layer 2 index: given the current user prompt,
    find the most relevant past turns.  The search is over
    each summary's ``search_text`` field (intent + symbols +
    headline + keywords).

    The index is small (one entry per turn, ~50 tokens each)
    and searches are sub-millisecond.
    """

    def __init__(self) -> None:
        self._summaries: dict[str, TurnSummary] = {}
        self._by_seq: list[TurnSummary] = []  # sorted by seq

        # Simple BM25 over search_text
        self._doc_tokens: dict[str, list[str]] = {}
        self._df: dict[str, int] = {}
        self._doc_lengths: list[int] = []
        self._avgdl: float = 0.0
        self._dirty: bool = False

    def insert(self, summary: TurnSummary) -> None:
        """Insert or replace a turn summary."""
        self._summaries[summary.turn_id] = summary

        # Rebuild the sorted list and BM25 data
        self._rebuild()

    def remove(self, turn_id: str) -> None:
        """Remove a turn summary."""
        if turn_id not in self._summaries:
            return
        del self._summaries[turn_id]
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild the BM25 data structures."""
        self._by_seq = sorted(self._summaries.values(), key=lambda s: s.seq)
        self._doc_tokens = {}
        self._df = {}
        self._doc_lengths = []

        for summary in self._by_seq:
            tokens = _tokenize_summary(summary.search_text)
            self._doc_tokens[summary.turn_id] = tokens
            self._doc_lengths.append(len(tokens))
            for t in set(tokens):
                self._df[t] = self._df.get(t, 0) + 1

        self._avgdl = sum(self._doc_lengths) / max(len(self._doc_lengths), 1)
        self._dirty = False

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        min_score: float = 0.01,
    ) -> list[tuple[TurnSummary, float]]:
        """Search for the most relevant past turns.

        Returns a list of (summary, score) pairs, highest first.
        """
        if not query or not query.strip() or not self._summaries:
            return []

        query_tokens = _tokenize_summary(query)
        if not query_tokens:
            return []

        n_docs = len(self._by_seq)
        scores: dict[str, float] = {}

        for turn_id, tokens in self._doc_tokens.items():
            if not tokens:
                continue
            dl = len(tokens)
            tf_counter: dict[str, int] = {}
            for t in tokens:
                tf_counter[t] = tf_counter.get(t, 0) + 1

            score = 0.0
            for qt in query_tokens:
                tf = tf_counter.get(qt, 0)
                if tf == 0:
                    continue
                n_qi = self._df.get(qt, 0)
                import math
                idf = math.log(1 + (n_docs - n_qi + 0.5) / (n_qi + 0.5))
                tf_norm = (tf * 2.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / max(self._avgdl, 1)))
                score += idf * tf_norm

            if score >= min_score:
                scores[turn_id] = score

        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            (self._summaries[tid], score)
            for tid, score in results
        ]

    def get(self, turn_id: str) -> TurnSummary | None:
        return self._summaries.get(turn_id)

    def get_recent(self, n: int) -> list[TurnSummary]:
        """Return the last N summaries by seq (most recent last)."""
        return self._by_seq[-n:] if n > 0 else []

    def clear(self) -> None:
        self._summaries.clear()
        self._by_seq.clear()
        self._doc_tokens.clear()
        self._df.clear()
        self._doc_lengths.clear()
        self._avgdl = 0.0

    def __len__(self) -> int:
        return len(self._summaries)

    def stats(self) -> dict[str, Any]:
        return {
            "n_summaries": len(self._summaries),
            "avgdl": self._avgdl,
            "vocab_size": len(self._df),
        }


def _tokenize_summary(text: str) -> list[str]:
    """Tokenize a summary's search text for BM25 indexing."""
    if not text:
        return []
    raw = _IDENTIFIER_RE.findall(text)
    return [t.lower() for t in raw if t.lower() not in _KEYWORD_STOPWORDS and len(t) > 2]


# ---------------------------------------------------------------------------
# Heuristic memory relevance gate
# ---------------------------------------------------------------------------

# Reference markers that signal the user is referring to prior turns
_REFERENCE_MARKERS = re.compile(
    r"\b(?:that|this|the\s+\w+\s+you|above|below|previous|last|just|"
    r"fix\s+it|continue|now\s+do|instead\s+of|try\s+again|"
    r"the\s+function|the\s+class|the\s+code|the\s+method|"
    r"what\s+you\s+(?:just\s+)?(?:wrote|said|generated|created|did)|"
    r"earlier|before\s+this|in\s+the\s+(?:previous|last|above))\b",
    re.IGNORECASE,
)

# Short prompts that are likely iterations
_SHORT_ITERATIONS = re.compile(
    r"^(?:yes|no|continue|try again|redo|fix it|go ahead|ok|sure|again|why\?|how\?|explain)\s*[.!?]*$",
    re.IGNORECASE,
)


def heuristic_memory_relevance(
    prompt: str,
    recent_summaries: list[TurnSummary],
    recent_chunks: list[Chunk],
) -> str:
    """Determine whether memory injection is needed.

    Returns one of:
    * ``"none"`` — clear new task, no injection needed
    * ``"last_turn"`` — likely iterating on the last turn
    * ``"specific"`` — references a specific prior chunk
    * ``"broad"`` — wants to continue prior work broadly
    * ``"cross_turn"`` — references a specific older turn
    * ``"ambiguous"`` — can't tell, inject last-turn summary as default

    This function is intentionally cheap (regex + set overlap).
    When it returns ``"ambiguous"``, the caller may invoke the
    router model for a higher-quality judgment.
    """
    if not prompt or not prompt.strip():
        return "none"

    prompt_lower = prompt.lower().strip()

    # Signal 1: short iteration prompts
    if _SHORT_ITERATIONS.match(prompt_lower):
        return "last_turn"

    # Signal 2: reference markers
    has_markers = bool(_REFERENCE_MARKERS.search(prompt))

    # Signal 3: symbol overlap with recent chunks
    prompt_identifiers = set(
        m.group(0).lower()
        for m in _IDENTIFIER_RE.finditer(prompt)
        if len(m.group(0)) > 2
    )
    recent_symbols = {c.symbol.lower() for c in recent_chunks if c.symbol}
    symbol_overlap = prompt_identifiers & recent_symbols

    # Signal 4: keyword overlap with recent turn summaries
    # Exclude very common words that cause false positives
    _COMMON_FALSE_POSITIVES = frozenset({
        "write", "function", "code", "python", "the", "task",
        "prompt", "problem", "solution", "create", "make", "add",
    })
    prompt_words = set(
        w.lower() for w in _IDENTIFIER_RE.findall(prompt)
        if w.lower() not in _KEYWORD_STOPWORDS and len(w) > 2
    )
    summary_keywords: set[str] = set()
    for s in recent_summaries:
        summary_keywords.update(k.lower() for k in s.keywords)
        summary_keywords.update(s.lower() for s in s.symbols)
    # Filter out common false positives
    keyword_overlap = (prompt_words & summary_keywords) - _COMMON_FALSE_POSITIVES

    # Decision logic: combine signals
    if has_markers and symbol_overlap:
        # User is referencing a specific prior artifact by name
        # Check if the symbol is from the most recent turn
        if recent_chunks:
            last_turn_chunks = [c for c in recent_chunks if c.turn_id == recent_chunks[-1].turn_id]
            last_turn_symbols = {c.symbol.lower() for c in last_turn_chunks if c.symbol}
            if symbol_overlap & last_turn_symbols:
                return "specific"
        # Otherwise it's from an older turn
        return "cross_turn"

    if has_markers and not symbol_overlap and not keyword_overlap:
        # References prior work but no specific match — broad
        return "broad"

    if not has_markers and symbol_overlap:
        # No explicit reference but mentions a prior symbol — specific
        return "specific"

    if not has_markers and not symbol_overlap and keyword_overlap:
        # Keyword overlap suggests related work — broad
        return "broad"

    if has_markers and keyword_overlap and not symbol_overlap:
        # Some connection but unclear — ambiguous
        return "ambiguous"

    # No markers, no overlap — likely a new task
    return "none"