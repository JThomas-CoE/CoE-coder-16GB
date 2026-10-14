"""Evaluate-object holding pen — single-slot persistence for evaluate_artifact.

When the evaluate_artifact harness runs, it stores BOTH the source object it
analyzed (an attached file, pasted content, or a stored artifact) AND the
work product (the evaluation report).  This lets a follow-up turn (e.g. "fix
all the bugs you found") operate on the same source context.

Design (per user spec):
* **Single-slot**: only one stored object at a time.  A NEW evaluate_artifact
  task replaces it.  modify/bug_fix/general_use do NOT clear it.
* **One-layer-deep**: the WP of a "fix the bugs" round is naturally an artifact
  produced by the modify_artifact harness, not another evaluate object.
* **Persists across turns AND server restarts** (disk-backed).
* **Slash-override friendly**: if a follow-up routes to bug_fix by mistake, the
  user can /task-modify_artifact (legacy /task_type-modify_artifact also
  accepted) and the correct evaluate object is still available.

The source object is whatever the user attached/pasted/selected — NOT
necessarily a stored artifact.  The store captures the actual object analyzed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class EvaluateObjectStore:
    """Single-slot holding pen for evaluate_artifact source + WP.

    Persists to ``evaluate_objects/`` under the package directory
    (mirroring ``artifacts_saved/``), so it survives server restarts.
    """

    def __init__(self, base_dir: str = "evaluate_objects") -> None:
        import pathlib
        pkg_dir = pathlib.Path(__file__).resolve().parent
        base = pathlib.Path(base_dir)
        self._dir = base if base.is_absolute() else pkg_dir / base
        self._dir.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] | None = None
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def put(
        self,
        source_content: str,
        source_language: str | None,
        wp_content: str,
        wp_language: str | None,
        source_filename: str | None = None,
    ) -> None:
        """Store the evaluate source object + its work product (single slot)."""
        self._state = {
            "source_content": source_content,
            "source_language": source_language,
            "source_filename": source_filename,
            "wp_content": wp_content,
            "wp_language": wp_language,
            "created_at": time.time(),
        }
        self._persist()

    def get(self) -> dict[str, Any] | None:
        """Return the stored {source_content, source_language, wp_content, ...} or None."""
        return self._state

    def has(self) -> bool:
        return self._state is not None

    def clear(self) -> None:
        """Clear the holding pen (used when a new evaluate task replaces it)."""
        self._state = None
        self._persist()

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------
    def _state_path(self) -> Path:
        return self._dir / "evaluate_object.json"

    def _persist(self) -> None:
        try:
            self._state_path().write_text(
                json.dumps(self._state, indent=2) if self._state else "null",
                encoding="utf-8",
            )
        except Exception:
            # Never break the inference on a disk hiccup.
            pass

    def _load(self) -> None:
        try:
            p = self._state_path()
            if p.exists():
                raw = p.read_text(encoding="utf-8")
                self._state = json.loads(raw) if raw.strip() else None
        except Exception:
            self._state = None
