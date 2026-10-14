"""Memory injection assembler for session memory V2.

Replaces the V1 ``build_last_turn_section()`` with a
multi-layer injection that assembles a labeled memory
section from:

1. **Chronological window** — the last N chunks (always
   present when injection is enabled)
2. **Semantic neighbors** — chunks from older turns that
   match the current prompt via BM25 (added if budget
   allows)

The assembler is budget-aware: it fills from the
chronological window first, then spends any remaining
budget on semantic matches.  The output is always clearly
labeled so the model knows this is context, not
instructions.

The injection format:

    --- SESSION MEMORY (injected for context) ---

    [CHRONOLOGICAL — last N chunks]
    Turn 42: <intent>
      <chunk text with boundary context>

    Turn 41: <intent>
      <chunk text with boundary context>

    [SEMANTIC MATCH — from turn 38]
    Turn 38: <intent>
      <matched chunk>

    --- END SESSION MEMORY ---

If the injection is empty (clear new task, ``relevance ==
"none"``), an empty string is returned — no wasted tokens.
"""
from __future__ import annotations

from typing import Any

from .chunker import Chunk
from .chunk_index import BM25Index
from .session_memory import estimate_tokens
from .turn_summary import TurnSummary, TurnSummaryIndex


# ---------------------------------------------------------------------------
# Budget constants
# ---------------------------------------------------------------------------

DEFAULT_BUDGET_TOKENS = 4000  # max tokens for memory injection section
MIN_CHUNKS_CHRONO = 1         # always include at least 1 recent chunk
MAX_CHUNKS_SEMANTIC = 3        # max semantic matches to inject
MIN_SCORE_THRESHOLD = 0.5     # BM25 min score for semantic matches


# ---------------------------------------------------------------------------
# Injection assembler
# ---------------------------------------------------------------------------

def build_memory_section(
    relevance: str,
    current_prompt: str,
    turn_summaries: TurnSummaryIndex,
    chunk_index: BM25Index,
    *,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    recent_turn_ids: list[str] | None = None,
) -> str:
    """Assemble a labeled memory section for prompt injection.

    Parameters
    ----------
    relevance
        The memory relevance verdict from
        ``heuristic_memory_relevance()``: ``"none"``,
        ``"last_turn"``, ``"specific"``, ``"broad"``,
        ``"cross_turn"``, or ``"ambiguous"``.
    current_prompt
        The current user prompt (used for semantic search).
    turn_summaries
        Layer 2 index: BM25 over turn summaries.
    chunk_index
        Layer 3 index: BM25 over chunks.
    budget_tokens
        Maximum tokens for the injection section.
    recent_turn_ids
        IDs of recent turns to include in the chronological
        window.  If None, uses the last 2 turns from the
        summary index.

    Returns
    -------
    str
        The labeled memory section, or empty string if
        no injection is needed.
    """
    if relevance == "none":
        return ""

    # Determine which turns to include chronologically
    if recent_turn_ids is None:
        recent = turn_summaries.get_recent(2)
        recent_turn_ids = [s.turn_id for s in recent]

    if not recent_turn_ids and relevance != "cross_turn":
        return ""

    parts: list[str] = [
        "\n\n--- SESSION MEMORY (injected for context, may not be relevant) ---\n",
    ]
    budget_remaining = budget_tokens

    # Layer 1: Chronological window
    chrono_chunks: list[Chunk] = []
    chrono_turns: list[TurnSummary] = []

    # For "last_turn" relevance, only include the most recent turn
    if relevance == "last_turn" and recent_turn_ids:
        turn_ids_to_use = recent_turn_ids[-1:]
    elif relevance in ("specific", "broad", "ambiguous") and recent_turn_ids:
        turn_ids_to_use = recent_turn_ids
    elif relevance == "cross_turn":
        # Don't include chronological, only semantic
        turn_ids_to_use = []
    else:
        turn_ids_to_use = recent_turn_ids or []

    for turn_id in reversed(turn_ids_to_use):
        summary = turn_summaries.get(turn_id)
        if not summary:
            continue
        turn_chunks = chunk_index.get_chunks_for_turn(turn_id)
        if not turn_chunks:
            continue

        # Include chunks from this turn, respecting budget
        turn_text_parts: list[str] = []
        turn_tokens = 0
        for chunk in turn_chunks:
            chunk_tokens = chunk.token_estimate
            if budget_remaining - chunk_tokens < 0:
                break
            turn_text_parts.append(_format_chunk(chunk))
            turn_tokens += chunk_tokens
            budget_remaining -= chunk_tokens
            chrono_chunks.append(chunk)

        if turn_text_parts:
            chrono_turns.append(summary)
            parts.append(f"\n[CHRONOLOGICAL — turn {summary.seq}]\n")
            parts.append(f"Turn {summary.seq}: {summary.intent}\n")
            parts.append("".join(turn_text_parts))

    # Layer 2+3: Semantic neighbors (for specific, cross_turn, broad)
    if relevance in ("specific", "cross_turn", "broad", "ambiguous"):
        semantic_chunks = _find_semantic_chunks(
            current_prompt, chunk_index, turn_summaries,
            exclude_turn_ids=set(turn_ids_to_use),
            budget_tokens=budget_remaining,
        )

        if semantic_chunks:
            parts.append("\n[SEMANTIC MATCH — from earlier turns]\n")
            for chunk, summary, score in semantic_chunks:
                parts.append(f"Turn {summary.seq} ({summary.intent}):\n")
                parts.append(_format_chunk(chunk))
                budget_remaining -= chunk.token_estimate

    parts.append("\n--- END SESSION MEMORY ---\n")

    # If nothing was injected, return empty
    if not chrono_chunks and not any("[SEMANTIC" in p for p in parts):
        return ""

    return "".join(parts)


def _format_chunk(chunk: Chunk) -> str:
    """Format a chunk for injection with boundary context."""
    parts: list[str] = []
    if chunk.prev_context:
        parts.append(f"  ...{chunk.prev_context.strip()[-200:]}\n")
    parts.append(f"  {chunk.text.strip()}\n")
    if chunk.next_context:
        parts.append(f"  {chunk.next_context.strip()[:200]}...\n")
    return "".join(parts)


def _find_semantic_chunks(
    query: str,
    chunk_index: BM25Index,
    turn_summaries: TurnSummaryIndex,
    exclude_turn_ids: set[str],
    budget_tokens: int,
) -> list[tuple[Chunk, TurnSummary, float]]:
    """Find semantic match chunks via BM25.

    Excludes chunks from turns already in the chronological
    window.  Returns up to MAX_CHUNKS_SEMANTIC chunks, each
    with its parent turn's summary and BM25 score.
    """
    if budget_tokens < 200:
        return []

    # Search the chunk index
    results = chunk_index.search(query, k=MAX_CHUNKS_SEMANTIC * 2, min_score=MIN_SCORE_THRESHOLD)
    if not results:
        return []

    # Filter: exclude turns already in chronological window
    filtered: list[tuple[Chunk, TurnSummary, float]] = []
    for chunk, score in results:
        if chunk.turn_id in exclude_turn_ids:
            continue
        summary = turn_summaries.get(chunk.turn_id)
        if not summary:
            continue
        filtered.append((chunk, summary, score))

    # Sort by score descending, take top MAX_CHUNKS_SEMANTIC
    filtered.sort(key=lambda x: x[2], reverse=True)
    return filtered[:MAX_CHUNKS_SEMANTIC]