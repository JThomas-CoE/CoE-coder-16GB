"""Artifact store — mutable working objects in system RAM.

The artifact store is the lightweight persistence framework that
gives the CoE-coder the benefit of state persistence internally
as if it had a much larger context window.

Each artifact is a mutable object stored in system RAM (not
VRAM, not KV cache).  It has:

* ``artifact_id`` — unique identifier (auto-generated or user-provided)
* ``content`` — the full text of the artifact (HTML, Python, prose, etc.)
* ``language`` — detected language (python, web, compiled_coding, prose)
* ``sections`` — tree-sitter-parsed addressable sections (functions, classes, etc.)
* ``created_at`` / ``modified_at`` — timestamps
* ``origin`` — how the artifact was created (generated, pasted, loaded)

The store supports:

* **create** — store a new artifact (from a generation turn or user paste)
* **get** — retrieve the full artifact content
* **get_section** — retrieve a specific section by name or BM25 match
* **update_section** — replace a section with new content
* **add_section** — add a new section to the artifact
* **list_sections** — list all sections with their names and sizes
* **search** — BM25 search over sections to find relevant pieces
* **switch** — set the "current" artifact (for multi-artifact sessions)
* **persist** — save versioned artifacts to disk
* **load** — hydrate bounded recent versions or load an exact catalog record

The store keeps a disk catalog of every saved version.  The newest
versions are hydrated into RAM up to a bounded content budget, while
older versions remain disk-backed and can be loaded exactly by catalog ID.

The artifact is never injected wholesale into a prompt.  Instead,
the relevant section(s) are extracted per turn and injected.  This
allows artifacts much larger than the context window to be worked
on piecewise.

The BM25 chunk index from ``clean_chunk_index.py`` is reused for
section-level search.  The tree-sitter chunker from
``clean_chunker.py`` is reused for section parsing.
"""
from __future__ import annotations

import pathlib
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .chunker import Chunk, chunk_model_output, detect_content_kind, detect_language
from .chunk_index import BM25Index
from .artifact_validation import validate_artifact_syntax


# ---------------------------------------------------------------------------
# Step 5: descriptor-slug helper
# ---------------------------------------------------------------------------

def _slugify(text: str, *, max_len: int = 100) -> str:
    """Turn free text into a filename/descriptor-safe slug.

    lowercase -> strip to [a-z0-9 -] -> collapse whitespace -> "-" join,
    truncated to ``max_len`` at a word boundary (never mid-word).
    Falls back to ``"untitled"`` for empty/all-punctuation input.

    ``max_len`` was raised 40 -> 100 (2026-09-01): the 40-char cap
    truncated descriptors before the topic word (e.g. a tic-tac-toe
    game became ``create-a-standalone-html-file-with``, losing
    "game"), which broke name-based artifact resolution.  100 chars
    keeps the full intent in the descriptor/filename while staying
    well under the Windows 255-char path-component limit.
    """
    s = text.lower()
    s = re.sub(r"[^a-z0-9 -]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "-")
    s = re.sub(r"-+", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rsplit("-", 1)[0]
    return s or "untitled"


# Leading CSS comment pattern for section-name normalization.
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _norm_section_name(name: str) -> str:
    """Normalize a section name for robust matching in ``apply_edits``.

    Strips leading ``/* ... */`` CSS comments and collapses whitespace
    so the model's edit target (e.g. ``.indicator-box``) matches a
    stored section name that might carry a comment prefix or extra
    whitespace.  The chunker already strips CSS comments from
    selectors, but this is a defensive measure — the model may also
    emit slight whitespace variations.
    """
    if not name:
        return name
    # Strip leading CSS comments (iteratively for multiple blocks).
    s = name
    while True:
        stripped = s.lstrip()
        if stripped.startswith("/*"):
            close = stripped.find("*/")
            if close == -1:
                break
            s = stripped[close + 2:].strip()
            continue
        break
    # Collapse internal whitespace to single spaces.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _synthetic_section_name(kind: str, char_start: int, char_end: int) -> str:
    """Return a stable address for a section without a source symbol."""
    safe_kind = re.sub(r"[^a-z0-9_]+", "_", (kind or "section").lower()).strip("_")
    return f"anonymous:{safe_kind or 'section'}:{char_start}-{char_end}"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ArtifactSection:
    """An addressable section within an artifact.

    A section is a chunk from the tree-sitter parse — a function,
    class, or prose block.  It has:

    * ``name`` — the function/class name (or None for prose)
    * ``kind`` — "code", "prose", "diff_hunk"
    * ``text`` — the section content
    * ``char_start`` / ``char_end`` — offsets into the artifact content
    * ``chunk`` — the underlying Chunk object (with boundary context)
    """
    name: str | None
    kind: str
    text: str
    char_start: int
    char_end: int
    chunk: Chunk | None = None


@dataclass
class Artifact:
    """A mutable working object stored in system RAM.

    The artifact is the user's current work product — an HTML
    file, a Python script, a prose document, etc.  It persists
    across turns and is modified in place, never regenerated
    wholesale (unless the user explicitly asks for a regeneration).

    Attributes
    ----------
    artifact_id
        Unique identifier (auto-generated or user-provided).
    content
        The full text of the artifact.
    language
        Detected language (python, web, compiled_coding, prose).
    sections
        Tree-sitter-parsed addressable sections.
    created_at
        Unix timestamp when the artifact was created.
    modified_at
        Unix timestamp when the artifact was last modified.
    origin
        How the artifact was created: "generated", "pasted",
        "loaded", or "forked".
    turn_id
        The turn that created this artifact (for session memory
        integration).
    """

    artifact_id: str
    content: str
    language: str | None = None
    sections: list[ArtifactSection] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    origin: str = "generated"
    turn_id: str | None = None
    produced_by_model: str | None = None
    # Domain provenance (2026-09-07): the routing domain that produced
    # this artifact (e.g. "domain_physics").  Used by the modify/evaluate
    # domain pin to prefer the PRODUCING domain over the language-map
    # heuristic (markdown/prose artifacts have no language->domain row,
    # which let a physics-domain artifact be reclassified domain_other).
    # None for external artifacts; "domain_other" is recorded but never
    # pins (only wired domains get pin power).
    produced_by_domain: str | None = None

    # --- Step 5: versioning/branching fields (added 2026-08-04) ---
    # parent_id: the artifact_id this version was forked from;
    #   None for a root (or an unpromoted staging entry).
    # session_number: lineage id, assigned once at root creation
    #   (or staging promotion) and inherited unchanged by every
    #   descendant in the lineage.  0 = not yet part of a lineage
    #   (unpromoted staging entry).
    # version: flat monotonic counter within the lineage.
    #   0 = unpromoted staging; 1 = root; 2+ = forked descendant.
    # descriptor: slug naming the lineage's topic (see _slugify);
    #   computed once at root creation/promotion, inherited
    #   unchanged by every descendant.  "" for staging entries.
    # source_filename: original filename stem, if this lineage's
    #   root/promotion came from an attachment with a filename.
    # saved_path: disk path this specific version was persisted
    #   to, or None if not yet persisted / persistence failed.
    parent_id: str | None = None
    session_number: int = 0
    version: int = 0
    descriptor: str = ""
    source_filename: str | None = None
    saved_path: str | None = None

    # --- Origin-prompt provenance (added 2026-09-14) ---
    # Supports the origin-gated modify pipeline
    # (CoE_coder_spec_modify_pipeline_DRAFT.md): a modification request on
    # an artifact that carries its originating prompt can be served by a
    # local regeneration from a synthesized combined prompt, avoiding the
    # targeted-editing change-order path the local specialists fail.
    #
    # origin_prompt: the request text this lineage was generated from.
    #   Assigned ONCE at root creation/promotion and inherited unchanged by
    #   every descendant -- it is NEVER rewritten to a combined prompt, so
    #   the full combo stays reconstructible as
    #   origin_prompt + [m["request"] for m in modifications] + current.
    #   None => the artifact has no known origin, so local regeneration is
    #   structurally unavailable and the pipeline falls back to the cloud
    #   change order.  None is the default so every existing code path and
    #   externally-loaded artifact behaves exactly as before.
    #
    # modifications: ordered, APPEND-ONLY log of modification requests
    #   applied to this lineage, oldest first.  Each entry records the
    #   request text plus how it was served, so provenance grows
    #   monotonically across cloud and local hops in any order.  Entries
    #   are never mutated or removed.
    origin_prompt: str | None = None
    modifications: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for debug/logging."""
        return {
            "artifact_id": self.artifact_id,
            "content_length": len(self.content),
            "language": self.language,
            "section_count": len(self.sections),
            "section_names": [s.name for s in self.sections if s.name],
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "origin": self.origin,
            "turn_id": self.turn_id,
            "produced_by_model": self.produced_by_model,
            "produced_by_domain": self.produced_by_domain,
            "parent_id": self.parent_id,
            "session_number": self.session_number,
            "version": self.version,
            "descriptor": self.descriptor,
            "source_filename": self.source_filename,
            "saved_path": self.saved_path,
            # Origin-prompt provenance.  The prompt text is deliberately NOT
            # included here (to_dict is used for debug/logging) -- only
            # presence and the modification count are reported, so a debug
            # dump never leaks the user's request text.
            "has_origin_prompt": self.origin_prompt is not None,
            "origin_prompt_len": len(self.origin_prompt or ""),
            "modification_count": len(self.modifications),
            "modification_methods": [m.get("method") for m in self.modifications],
        }


@dataclass(frozen=True)
class PersistedArtifactRecord:
    """Metadata for one saved artifact version on disk."""

    artifact_id: str
    path: pathlib.Path
    source_filename: str
    descriptor: str
    session_number: int
    version: int
    language: str | None
    size_bytes: int
    modified_at: float


# ---------------------------------------------------------------------------
# Step 5: disk persistence
# ---------------------------------------------------------------------------

_EXTENSION_TABLE: dict[str, str] = {
    "python": ".py", "cpp": ".cpp", "c": ".c", "rust": ".rs", "go": ".go",
    "java": ".java", "kotlin": ".kt", "scala": ".scala", "html": ".html",
    "javascript": ".js", "typescript": ".ts", "css": ".css", "jsx": ".jsx",
    "tsx": ".tsx", "sql": ".sql", "bash": ".sh", "json": ".json",
    "markdown": ".md", "md": ".md",
    # 2026-09-07: the CLI's attach loader maps .html/.js/.css -> "web"
    # and .md/.txt -> "prose"; these tags round-trip through the store
    # and must save with the right extension (a promoted fork of an
    # attached .html artifact saved as .txt before this fix).
    #
    # 2026-09-16: the loader NO LONGER emits the lossy "web" bucket (it sent
    # .js and .css there too, and this table can only map a tag to ONE
    # extension -- so a .js attachment saved as .html).  The loader now emits
    # the real language.  "web"/"code" are RETAINED because external clients
    # may still send them, and because artefacts already on disk carry them.
    "web": ".html", "prose": ".md", "code": ".txt",
    # 2026-09-16: alternates and same-family spellings that had no entry and
    # so fell to the ".txt" default.  These are reachable as a language tag
    # when a client supplies one directly (e.g. /v1/artifact/load).
    "htm": ".html", "xhtml": ".html",
    "svg": ".svg", "mjs": ".js", "cjs": ".js", "scss": ".css",
    # 2026-09-16: content-KINDS that can arrive here as a fallback tag when
    # language detection is not confident (see create()).  Unified diff has no
    # registered extension and no comment syntax of its own -- `.txt` is the
    # correct file for it, and it frames as "plain", so the header stays
    # readable AND strippable.  Declared explicitly so the mapping is not
    # inherited from _extension_for()'s `.txt` catch-all by accident.
    #
    # This table holds ONE flat namespace on purpose: its job is
    # "tag -> file extension", and language tags and content-kind tags both
    # need to answer that question.  They are different concepts
    # (chunker.detect_language vs chunker.detect_content_kind) that share a
    # key space because they resolve to the same thing here.
    "diff": ".txt",
}


def _extension_for(language: str | None) -> str:
    """Map a language tag to a file extension; unknown -> ``.txt``."""
    return _EXTENSION_TABLE.get((language or "").lower(), ".txt")


def _is_sidecar_file(path: Any) -> bool:
    """True for a provenance side-car (``<stem>.meta.json``).

    2026-09-14 FIX: the disk catalog matched side-cars as ARTIFACTS.  A
    side-car shares its artifact's stem (deliberately -- it must sit beside
    it), and the catalog's filename regex only anchors the PREFIX
    ``<session>_<descriptor>_v<version>_<timestamp>``, so
    ``..._v1_20260915T114953.meta.json`` matched and was catalogued as a
    second artifact with the SAME session/version/descriptor as the real
    file.

    Consequences of that single miss were three visible defects:
      1. the artifact picker showed every artifact TWICE (once as ``.py``,
         once as ``.meta.json``) -- in the live session, 5 of 10 choices
         were side-cars;
      2. ``.meta.json`` has no entry in the extension table, so it fell back
         to ``.txt`` and every side-car was HYDRATED AS A TEXT ARTIFACT,
         putting JSON provenance into the BM25 section index that the cloud
         is shown;
      3. because a side-car shares its artifact's session and version, the
         ``_disk_lineages`` bookkeeping counted the same lineage twice.

    A side-car is provenance METADATA, never an artifact: it is never a
    user-selectable target and must never be a model-call target.
    """
    try:
        return str(getattr(path, "name", path)).lower().endswith(".meta.json")
    except Exception:                                      # noqa: BLE001
        return False


def _strip_embedded_header(content: str, language: str | None) -> tuple[str, dict]:
    """Remove an embedded provenance header (2026-09-14).

    Soft-fails to (content, empty) whenever the provenance module is missing or
    anything goes wrong, so this can never break artifact creation.  The
    provenance module is OPTIONAL by design: deleting it reverts the whole
    origin-prompt pipeline without touching the store.
    """
    try:
        from .provenance import strip_header as _sh
    except ImportError:
        return content, {"origin_prompt": None, "modifications": []}
    try:
        return _sh(content, _extension_for(language))
    except Exception:                                      # noqa: BLE001
        return content, {"origin_prompt": None, "modifications": []}


def _attach_embedded_header(content: str, artifact: "Artifact") -> str:
    """Write the provenance header into the text that goes to disk.

    Only when the artifact has provenance to record.  Returns ``content``
    unchanged on any failure, and reports (via the caller) when the header could
    NOT be framed -- an unframed header means the extension is unknown, which is
    a DEFECT SIGNAL (language detection already failed), not a normal state.
    """
    origin_prompt = getattr(artifact, "origin_prompt", None)
    modifications = getattr(artifact, "modifications", None)
    if not origin_prompt:
        return content
    try:
        from .provenance import attach_header as _ah
    except ImportError:
        return content
    try:
        framed, info = _ah(content, origin_prompt, modifications,
                           _extension_for(artifact.language))
        if not info.get("framed"):
            print(f"[provenance] header NOT framed for language="
                  f"{artifact.language!r}: {info.get('reason')}",
                  file=sys.stderr, flush=True)
        return framed
    except Exception as _exc:                              # noqa: BLE001
        print(f"[provenance] header attach failed: {_exc}",
              file=sys.stderr, flush=True)
        return content


def save_artifact_to_disk(
    artifact: "Artifact",
    *,
    base_dir: str = "artifacts_saved",
) -> str | None:
    """Persist one artifact version to disk.

    Filename: ``<session_number>_<descriptor>_v<version>_<timestamp>.<ext>``.

    ``base_dir`` is resolved relative to the CoE-coder-16GB PACKAGE
    directory (``Path(__file__).parent``), NOT the process CWD.  This
    makes the save location deterministic regardless of where the
    server is launched from -- artifacts always land in
    ``CoE_agent/CoE_coder_16GB/artifacts_saved/``.  (Previously it was
    CWD-relative, which scattered artifacts into whatever folder the
    server happened to be started from.)

    This function never raises.  A write failure (permissions,
    disk full, unwritable path) is caught, ``artifact.saved_path``
    is left/set to ``None``, and ``None`` is returned.  Callers use
    a ``None`` return (or ``artifact.saved_path is None``) to fall
    back to returning full content instead of a disk-relative diff
    (spec 6.2's persistence-failure fallback).
    """
    import datetime
    import pathlib
    import sys as _sys

    try:
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ext = _extension_for(artifact.language)
        filename = f"{artifact.session_number}_{artifact.descriptor}_v{artifact.version}_{ts}{ext}"
        # Resolve relative to the package directory (deterministic),
        # not the process CWD.  An absolute base_dir is used as-is.
        pkg_dir = pathlib.Path(__file__).resolve().parent
        base = pathlib.Path(base_dir)
        out_dir = base if base.is_absolute() else pkg_dir / base
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        # Provenance header goes into the PERSISTED TEXT only.  The in-memory
        # artifact.content is ALWAYS the header-free body, so section indexing
        # and every model call see the artifact alone -- that is what prevents
        # the cloud from anchoring on or editing a provenance section.
        _persist_text = _attach_embedded_header(artifact.content, artifact)
        out_path.write_text(_persist_text, encoding="utf-8")
        artifact.saved_path = str(out_path)
        # 2026-09-14 (spec 2.5 / 7.3): the SIDE-CAR.  Written next to the
        # artifact sharing its stem.  Two jobs:
        #   1. it carries the BULKY `change_order` records that are
        #      deliberately kept OUT of the file header (a cloud change
        #      order can approach the artifact's own size, and the header
        #      must stay readable by a human opening the file), and
        #   2. it restates the short fields so a reader can learn an
        #      artifact's provenance from ONE small file without parsing
        #      the artifact itself.
        # On disagreement between the two, THE FILE HEADER WINS: it is the
        # copy that travels with the artifact when the file is copied out of
        # the folder, so it is the one that cannot go stale.
        _write_sidecar_meta(out_path, artifact)
        return artifact.saved_path
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] save_artifact_to_disk failed: {exc}", file=_sys.stderr)
        artifact.saved_path = None
        return None


def _write_sidecar_meta(out_path: Any, artifact: "Artifact") -> str | None:
    """Write ``<same stem>.meta.json`` beside the artifact (spec 2.5).

    NEVER written for a ``version=0`` staging entry: version 0 means
    "attached/loaded, unpromoted", so the artifact is not yet part of a
    lineage and has nothing to describe.  Writing one would leave an orphan
    side-car for every file a user merely pasted.

    Shares the artifact's stem exactly -- ``..._v2_20260914T101500.html``
    gets ``..._v2_20260914T101500.meta.json`` -- so a directory listing keeps
    the pair adjacent and a deletion of one is visually obvious.

    Never raises: a side-car failure must not fail the artifact save, since
    the artifact itself is the product and the side-car is an index over it.
    """
    import json
    import pathlib
    import sys as _sys

    if int(getattr(artifact, "version", 0) or 0) <= 0:
        return None
    try:
        _p = pathlib.Path(out_path)
        meta_path = _p.with_suffix(".meta.json")
        # `basis` is the same label the response metadata reports, so the
        # side-car and the live routing telemetry cannot drift apart.
        try:
            from .provenance import basis_label as _bl
            _basis = _bl(
                getattr(artifact, "origin_prompt", None),
                getattr(artifact, "modifications", None),
            )
        except Exception:
            _basis = None
        payload = {
            "schema": "coe-artifact-meta v1",
            "artifact_id": artifact.artifact_id,
            "session_number": artifact.session_number,
            "version": artifact.version,
            "parent_id": artifact.parent_id,
            "descriptor": artifact.descriptor,
            "language": artifact.language,
            "origin": artifact.origin,
            "origin_prompt": getattr(artifact, "origin_prompt", None),
            # FULL log, including any `change_order`.  This is the whole
            # point of the side-car: the header deliberately carries only
            # the short fields.
            "modifications": list(getattr(artifact, "modifications", None) or []),
            "basis": _basis,
            # Stated in the file itself so a reader who finds the side-car
            # alone knows how to resolve a conflict.
            "precedence_note": (
                "On disagreement, the ARTIFACT FILE HEADER wins -- it "
                "travels with the artifact when the file is copied elsewhere."
            ),
            "artifact_filename": _p.name,
        }
        meta_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(meta_path)
    except Exception as exc:  # noqa: BLE001 - side-car is strictly advisory
        print(f"[warn] sidecar meta write failed: {exc}", file=_sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Step 6.2: bounded diff-response format
# ---------------------------------------------------------------------------

def _context_window(lines: list[str], start_idx: int, direction: int) -> list[str]:
    """Walk outward from ``start_idx`` in ``direction`` (+1 forward,
    -1 backward), skipping blank/whitespace-only lines, accumulating
    until >=3 non-blank lines AND >=50 non-whitespace chars, capped
    at 150 non-whitespace chars regardless of line count.

    The 150-char cap is a soft cap on the running total: it stops
    accumulating once crossed, but a single very long line is never
    truncated mid-line, so the total may modestly exceed 150 for that
    one line (documented behavior, not a hard truncation).
    """
    collected: list[str] = []
    non_ws_chars = 0
    i = start_idx
    while 0 <= i < len(lines):
        line = lines[i]
        if line.strip():
            collected.append(line)
            non_ws_chars += len(line.strip())
            if non_ws_chars >= 150:
                break
            if len(collected) >= 3 and non_ws_chars >= 50:
                break
        i += direction
    if direction < 0:
        collected.reverse()
    return collected


def build_diff_response(
    parent: "Artifact | None",
    child: "Artifact",
    edit_metadata: dict[str, Any],
) -> str:
    """Render the bounded-context diff-hunk format for a successful
    modify_artifact commit.

    Hunks (both replacements and insertions) are ordered by the
    CHILD's own section position (``char_start``), not by the order
    they appear in ``edit_metadata``'s lists -- every section,
    replaced or newly inserted, has a real final position in the one
    artifact actually shown to the user, so sorting on that position
    (rather than trying to reconcile parent/child coordinate spaces)
    gives the correct "artifact order" regardless of how the edits
    were listed.
    """
    entries: list[tuple[int, str]] = []
    parent_by_name = {s.name: s for s in (parent.sections if parent else []) if s.name}
    child_by_name = {s.name: s for s in child.sections if s.name}
    child_lines = child.content.splitlines()

    for name in edit_metadata.get("sections_replaced", []):
        new_section = child_by_name.get(name)
        if not new_section:
            continue
        old_section = parent_by_name.get(name)
        start_line = child.content[:new_section.char_start].count("\n")
        end_line = child.content[:new_section.char_end].count("\n")
        before_ctx = _context_window(child_lines, start_line - 1, -1)
        after_ctx = _context_window(child_lines, end_line + 1, +1)
        hunk = [
            f"--- {child.descriptor} v{parent.version if parent else '?'} ({parent.artifact_id if parent else '?'})",
            f"+++ {child.descriptor} v{child.version} ({child.artifact_id})",
            f"@@ {name} @@",
        ]
        hunk += [f"  {l}" for l in before_ctx]
        if old_section:
            hunk += [f"- {l}" for l in old_section.text.splitlines()]
        hunk += [f"+ {l}" for l in new_section.text.splitlines()]
        hunk += [f"  {l}" for l in after_ctx]
        entries.append((new_section.char_start, "\n".join(hunk)))

    for name in edit_metadata.get("sections_added", []):
        new_section = child_by_name.get(name)
        if not new_section:
            continue
        start_line = child.content[:new_section.char_start].count("\n")
        end_line = child.content[:new_section.char_end].count("\n")
        before_ctx = _context_window(child_lines, start_line - 1, -1)
        after_ctx = _context_window(child_lines, end_line + 1, +1)
        hunk = [
            f"+++ {child.descriptor} v{child.version} ({child.artifact_id})",
            f"@@ {name} (added) @@",
        ]
        hunk += [f"  {l}" for l in before_ctx]
        hunk += [f"+ {l}" for l in new_section.text.splitlines()]
        hunk += [f"  {l}" for l in after_ctx]
        entries.append((new_section.char_start, "\n".join(hunk)))

    entries.sort(key=lambda e: e[0])  # artifact order, not edit-list order
    return "\n\n".join(text for _, text in entries) if entries else child.content


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------

class ArtifactStore:
    """Manages working artifacts in system RAM.

    The store holds one or more artifacts.  One artifact is
    designated as the "current" artifact — the one the user is
    actively working on.  When the user starts a new task, a new
    artifact is created and becomes current.  When the user
    returns to a previous artifact, it becomes current again.

    The store is designed to be lightweight: no file system, no
    git, no cursor tracking.  Just: store, extract, modify, persist.

    Parameters
    ----------
    max_artifacts
        Maximum number of artifacts to hold in RAM.  Older
        artifacts are evicted when the limit is reached.
        Default: 10.
    """

    _DISK_HYDRATION_BUDGET_BYTES = 10 * 1024 * 1024

    def __init__(self, *, max_artifacts: int = 10) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._current_id: str | None = None
        self._max_artifacts = max_artifacts
        self._id_counter: int = 0
        self._chunk_index = BM25Index()
        # --- Step 5: versioning counters ---
        # _session_counter assigns a fresh session_number to every
        # new root (denovo generation or staging promotion).
        # _lineage_version_counter tracks the highest version issued
        # so far per session_number for disk continuity. True forks
        # use parent-relative versions so sibling A/B branches from
        # v1 are both v2.
        self._session_counter: int = 0
        self._lineage_version_counter: dict[int, int] = {}
        self._disk_catalog: dict[str, PersistedArtifactRecord] = {}
        self._disk_hydrated_bytes = 0
        # --- Disk-lineage index (2026-08-14) ---
        # Maps descriptor -> (session_number, max_version) for every
        # lineage found on disk.  Built at init by scanning
        # artifacts_saved/.  Used by apply_edits() to reuse an existing
        # lineage when a loaded artifact is modified for the first time
        # (so the version continues from where the disk left off, not
        # restarted at v1 on every server restart).
        self._disk_lineages: dict[str, tuple[int, int]] = {}
        self._scan_disk_lineages()

    # ------------------------------------------------------------------
    # Disk-lineage scanning
    # ------------------------------------------------------------------

    _FILENAME_RE = re.compile(
        r"^(\d+)_(.+)_v(\d+)_\d{8}T\d{6}",
        re.IGNORECASE,
    )

    def _scan_disk_lineages(self) -> None:
        """Scan the ``artifacts_saved/`` folder and rebuild lineage
        state from on-disk filenames.

        Filenames follow the pattern::

            <session>_<descriptor>_v<version>_<timestamp>.<ext>

        where ``descriptor`` uses hyphens (from ``_slugify``), never
        underscores, so the ``_v`` delimiter is unambiguous.

        After this scan, ``_session_counter`` is the highest session
        number seen, ``_lineage_version_counter`` has the max version
        per session, and ``_disk_lineages`` maps descriptor ->
        (session_number, max_version) so a subsequent promotion of
        a loaded artifact with the same descriptor reuses the lineage.
        """
        import pathlib

        try:
            pkg_dir = pathlib.Path(__file__).resolve().parent
            art_dir = pkg_dir / "artifacts_saved"
            if not art_dir.is_dir():
                return
            for entry in art_dir.iterdir():
                if not entry.is_file():
                    continue
                # 2026-09-14: side-cars are provenance metadata, never
                # artifacts -- see _is_sidecar_file().  Without this skip the
                # catalog gains a duplicate entry for every saved version.
                if _is_sidecar_file(entry):
                    continue
                m = self._FILENAME_RE.match(entry.stem)
                if not m:
                    continue
                session_num = int(m.group(1))
                descriptor = m.group(2)
                version = int(m.group(3))
                stat = entry.stat()
                artifact_id = f"disk:{entry.name}"
                language = {
                    ".html": "html",
                    ".htm": "html",
                    ".js": "javascript",
                    ".ts": "typescript",
                    ".css": "css",
                    ".py": "python",
                    ".cpp": "cpp",
                    ".c": "c",
                    ".rs": "rust",
                    ".go": "go",
                    ".java": "java",
                    ".kt": "kotlin",
                    ".scala": "scala",
                }.get(entry.suffix.lower())
                self._disk_catalog[artifact_id] = PersistedArtifactRecord(
                    artifact_id=artifact_id,
                    path=entry.resolve(),
                    source_filename=entry.name,
                    descriptor=descriptor,
                    session_number=session_num,
                    version=version,
                    language=language,
                    size_bytes=stat.st_size,
                    modified_at=stat.st_mtime,
                )
                self._session_counter = max(self._session_counter, session_num)
                self._lineage_version_counter[session_num] = max(
                    self._lineage_version_counter.get(session_num, 0),
                    version,
                )
                existing = self._disk_lineages.get(descriptor)
                if existing is None or version > existing[1]:
                    self._disk_lineages[descriptor] = (session_num, version)
        except Exception:
            # Never crash the server at boot over a disk scan.
            # The in-memory counters fall back to 0/empty, which means
            # new lineages start at v1 (the pre-fix behavior).
            pass

        self._hydrate_disk_artifacts()

    def _hydrate_disk_artifacts(self) -> None:
        """Load newest saved versions into RAM up to the content budget.

        Every saved version remains in ``_disk_catalog``.  Only the newest
        records that fit within the 10 MiB content budget are parsed and
        indexed at startup.  Hydrated artifacts are deliberately not made
        current; explicit name/version selection must choose one.
        """
        records = sorted(
            self._disk_catalog.values(),
            key=lambda item: (item.modified_at, item.session_number, item.version),
            reverse=True,
        )
        for record in records:
            if self._disk_hydrated_bytes + record.size_bytes > self._DISK_HYDRATION_BUDGET_BYTES:
                continue
            if self._load_persisted_record(record, make_current=False) is not None:
                self._disk_hydrated_bytes += record.size_bytes
        self._current_id = None

    def _load_persisted_record(
        self, record: PersistedArtifactRecord, *, make_current: bool,
    ) -> Artifact | None:
        """Load one exact catalog record into RAM and index its sections."""
        existing = self._artifacts.get(record.artifact_id)
        if existing is not None:
            if make_current:
                self._current_id = record.artifact_id
            return existing
        try:
            artifact = Artifact(
                artifact_id=record.artifact_id,
                content=record.path.read_text(encoding="utf-8"),
                language=record.language,
                origin="loaded",
                created_at=record.modified_at,
                modified_at=record.modified_at,
                session_number=record.session_number,
                version=record.version,
                descriptor=record.descriptor,
                source_filename=record.source_filename,
                saved_path=str(record.path),
            )
            self._parse_sections(artifact)
            self._artifacts[record.artifact_id] = artifact
            self._index_artifact(artifact)
            if make_current:
                self._current_id = record.artifact_id
            return artifact
        except (OSError, UnicodeError):
            return None

    def load_persisted(self, artifact_id: str, *, make_current: bool = True) -> Artifact | None:
        """Load an exact saved version by catalog ID."""
        record = self._disk_catalog.get(artifact_id)
        if record is None:
            return None
        was_loaded = artifact_id in self._artifacts
        artifact = self._load_persisted_record(record, make_current=make_current)
        if artifact is not None and not was_loaded:
            self._disk_hydrated_bytes += record.size_bytes
            self._evict_if_needed()
        return artifact

    # ------------------------------------------------------------------
    # Artifact lifecycle
    # ------------------------------------------------------------------

    def create(
        self,
        content: str,
        *,
        language: str | None = None,
        origin: str = "generated",
        turn_id: str | None = None,
        artifact_id: str | None = None,
        produced_by_model: str | None = None,
        produced_by_domain: str | None = None,
        descriptor: str = "",
        source_filename: str | None = None,
        origin_prompt: str | None = None,
        modifications: list[dict[str, Any]] | None = None,
        descendant_of: "Artifact | None" = None,
        user_intent: str = "",
    ) -> str:
        """Create a new artifact and set it as current.

        Parameters
        ----------
        content
            The full text of the artifact.
        language
            Optional language hint (python, web, compiled_coding).
            If not provided, will be auto-detected.
        origin
            How the artifact was created.
        turn_id
            The turn that created this artifact.
        artifact_id
            Optional custom ID.  If not provided, auto-generated.
        produced_by_model
            The model ID that generated this artifact (for
            internal artifacts).  None for external artifacts
            (pasted/loaded).  Used by the edit pipeline to
            route modify/evaluate turns to the same model.
        produced_by_domain
            The routing domain that produced this artifact
            (e.g. "domain_physics").  Feeds the modify/evaluate
            domain pin; only wired domains pin, "domain_other"
            never does.  None for external artifacts.
        descriptor
            Step 5: slug naming the lineage's topic.  Only
            honored for ``origin="generated"`` (denovo root
            creation) -- ``"pasted"``/``"loaded"`` content is
            always an unpromoted staging entry (version=0),
            per spec 5.4.2 case 2; the descriptor for that case
            is assigned later, at promotion time.
        source_filename
            Step 5: the attachment's original filename stem, if
            any.  Recorded even on staging entries so a later
            promotion (apply_edits()) can use it for the
            descriptor.
        origin_prompt
            2026-09-14: the request text this lineage was
            generated from, if known.  None (the default) means
            the origin is unknown, which makes local
            regeneration unavailable for this lineage and routes
            any modification through the cloud change order.
            Only meaningful for ``origin="generated"`` roots and
            for externally-loaded artifacts whose caller supplies
            it; inherited unchanged by every descendant.
        descendant_of
            2026-09-14: the artifact this one is a MODIFIED VERSION of
            (PATH A regeneration).  When given, no new lineage is minted:
            the naming rules MIRROR ``apply_edits()`` exactly, so the same
            modification gets the same session/descriptor whichever route
            served it -- promote when the parent is an unpromoted staging
            entry (version 0), fork when it is a real version.
        user_intent
            2026-09-14: the router's one-line restatement of the request.
            Used ONLY for the promotion descriptor (``modify-<file>-<intent>``),
            matching apply_edits().  Ignored in every other case.

        Returns
        -------
        str
            The artifact ID.
        """
        if artifact_id is None:
            self._id_counter += 1
            artifact_id = f"artifact_{self._id_counter}"

        # --- Language detection BEFORE the header strip (2026-09-16) ---
        # ORDER IS LOAD-BEARING.  The strip below frames the header by
        # EXTENSION, and the extension comes from the language.  If detection
        # has not run yet then `language` is whatever the caller passed -- for
        # `/v1/artifact/load` that is None, so `_extension_for(None)` returned
        # ".txt", whose framing is "plain", which cannot match an HTML comment.
        # The strip then silently found nothing: provenance was lost AND the
        # inert header stayed in the body as content.  Detecting first gives
        # the strip the real language so it can frame-match correctly.
        #
        # This must yield a LANGUAGE tag, never a content KIND, because every
        # consumer downstream keys on it as a language: `_extension_for` picks
        # the file extension and `provenance` picks the framing from that.
        #
        # The previous version wrote `detect_content_kind()`'s answer straight
        # into this field.  For an attached HTML document that answer is
        # `"prose"`, so the artifact saved as `.md` AND its header could never
        # be stripped.  `detect_language` returns None when not confident, and
        # a WRONG tag is more damaging than an honest absence (wrong
        # extension, wrong framing, wrong domain pin), so the fallback is the
        # content kind's declared extension and nothing more.
        if language is None:
            language = detect_language(content)
            if language is None:
                _kind = detect_content_kind(content, language)
                language = "prose" if _kind == "mixed" else _kind

        # --- Origin-prompt header: STRIP BEFORE ANYTHING ELSE (2026-09-14) ---
        # A file arriving with an embedded header must have it removed before
        # section parsing, BM25 indexing, or any model call.  If the header
        # reached the index, the cloud could anchor on or edit a provenance
        # section, and the outline would gain a section that is not part of the
        # artifact.  Stripping here also recovers the provenance for a file
        # that was round-tripped through disk, so the pipeline can keep working
        # on an artifact it did not create in this process.
        if isinstance(content, str):
            _stripped, _hdr = _strip_embedded_header(content, language)
            if _stripped != content:
                content = _stripped
                if origin_prompt is None and _hdr.get("origin_prompt"):
                    origin_prompt = _hdr["origin_prompt"]
                # Recover the FULL log, not just the origin prompt: a reloaded
                # artifact must keep its modification history or the next
                # regeneration would drop every prior change (the V9 failure).
                if not modifications and _hdr.get("modifications"):
                    modifications = _hdr["modifications"]

        # --- Step 5: root provenance vs. unpromoted staging ---
        if descendant_of is not None:
            # 2026-09-14 (spec 5.7.2 PATH A): a MODIFIED VERSION of an
            # existing lineage, created without apply_edits' splice
            # bookkeeping because a regeneration produces whole content rather
            # than a change order.  It must NOT mint a new lineage the way
            # origin="generated" does -- that would burn a session number and
            # register a stale _disk_lineages entry, corrupting the version
            # counter for the real lineage.
            #
            # The naming RULES below deliberately MIRROR apply_edits() exactly,
            # because a regeneration and a change order are two ways of making
            # the SAME kind of product.  If they diverged, the same artifact
            # would get a different session/descriptor depending on which route
            # served the modification, and the version chain would fork:
            #   * parent is an UNPROMOTED staging entry (version == 0) ->
            #     PROMOTE: fresh descriptor `modify-<file>-<intent>`, a new
            #     session number (or the existing one when a lineage with that
            #     descriptor already exists on disk), version=1, and
            #     parent_id=None (the staging entry was never surfaced as a
            #     version, so it is not linked as a parent).
            #   * parent is a real version (version >= 1) -> FORK: inherit
            #     session_number/descriptor/source_filename unchanged,
            #     version = parent.version + 1, parent_id = parent id.
            # Both always mint a NEW artifact_id.
            if int(getattr(descendant_of, "version", 0) or 0) == 0:
                _eff_fn = source_filename or getattr(descendant_of, "source_filename", None)
                if _eff_fn:
                    _fn_slug = _slugify(
                        pathlib.Path(str(_eff_fn)).stem
                    )
                    _in_slug = _slugify(user_intent) if user_intent else ""
                    descriptor = (
                        f"modify-{_fn_slug}-{_in_slug}" if _in_slug
                        else f"modify-{_fn_slug}"
                    )
                else:
                    descriptor = "modify-" + _slugify(user_intent or "untitled")
                # Disk-lineage continuity, same as apply_edits: if a lineage
                # with this descriptor already exists on disk, continue it
                # rather than starting a fresh v1 (survives restarts).
                _existing = self._disk_lineages.get(descriptor)
                if _existing is not None:
                    _sess, _maxv = _existing
                    session_number = _sess
                    version = _maxv + 1
                    self._lineage_version_counter[_sess] = version
                    self._disk_lineages[descriptor] = (_sess, version)
                else:
                    self._session_counter += 1
                    session_number = self._session_counter
                    version = 1
                    self._lineage_version_counter[session_number] = 1
                    self._disk_lineages[descriptor] = (session_number, 1)
            else:
                session_number = descendant_of.session_number or 0
                descriptor = descendant_of.descriptor or descriptor
                source_filename = source_filename or descendant_of.source_filename
                version = int(descendant_of.version) + 1
                if session_number:
                    self._lineage_version_counter[session_number] = max(
                        self._lineage_version_counter.get(session_number, 0),
                        version,
                    )
                    if descriptor:
                        self._disk_lineages[descriptor] = (
                            session_number,
                            max(self._disk_lineages.get(descriptor, (0, 0))[1], version),
                        )
        elif origin == "generated":
            self._session_counter += 1
            session_number = self._session_counter
            version = 1
            self._lineage_version_counter[session_number] = 1
            # Register the new lineage so a subsequent fork
            # (apply_edits on version >= 1) can find it.
            if descriptor:
                self._disk_lineages[descriptor] = (session_number, 1)
        else:
            # pasted / loaded: unpromoted staging entry (5.4.2 case 2).
            # descriptor is intentionally ignored here -- it is only
            # ever assigned at promotion time (apply_edits()).
            session_number = 0
            version = 0
            descriptor = ""

        artifact = Artifact(
            artifact_id=artifact_id,
            content=content,
            language=language,
            # Origin provenance follows apply_edits EXACTLY:
            #   FORK  (parent was a real version) -> "forked"
            #   PROMOTE (parent was unpromoted staging) -> the STAGING
            #           artifact's own origin ("pasted"/"loaded")
            # Setting "forked" unconditionally would mislabel a promotion: a
            # promoted artifact descends from a user-supplied file, not from a
            # prior version of a lineage, and a listing that said "forked" would
            # claim a parent lineage that never existed.
            origin=("forked"
                    if int(getattr(descendant_of, "version", 0) or 0) >= 1
                    else (getattr(descendant_of, "origin", None) or origin)),
            turn_id=turn_id,
            produced_by_model=produced_by_model,
            produced_by_domain=produced_by_domain,
            parent_id=(
                None if (descendant_of is not None
                         and int(getattr(descendant_of, "version", 0) or 0) == 0)
                else (descendant_of.artifact_id if descendant_of is not None else None)
            ),
            session_number=session_number,
            version=version,
            descriptor=descriptor,
            source_filename=source_filename,
            origin_prompt=origin_prompt,
            modifications=list(modifications or []),
        )

        # Parse into sections
        self._parse_sections(artifact)

        # Store
        self._artifacts[artifact_id] = artifact
        self._current_id = artifact_id

        # Index sections in BM25
        self._index_artifact(artifact)

        # Evict if over limit
        self._evict_if_needed()

        return artifact_id

    # NOTE (2026-09-14): `next_lineage_version()` was introduced for the PATH A
    # regeneration commit and then REMOVED once the version bump moved into
    # ``create(descendant_of=...)``, which mirrors apply_edits' naming rules
    # directly.  A full-tree search confirmed it had no callers, so keeping it
    # would have left a second, divergent version-allocation path next to the
    # real one -- exactly the kind of dead-but-plausible helper a future change
    # would pick up by mistake.  Version allocation has ONE owner:
    # ``create()``'s descendant branch.

    def get(self, artifact_id: str | None = None) -> Artifact | None:
        """Get an artifact by ID, or the current artifact if ID is None."""
        if artifact_id is None:
            artifact_id = self._current_id
        if artifact_id is None:
            return None
        return self._artifacts.get(artifact_id)

    def get_content(self, artifact_id: str | None = None) -> str:
        """Get the full content of an artifact."""
        artifact = self.get(artifact_id)
        return artifact.content if artifact else ""

    def switch(self, artifact_id: str) -> bool:
        """Set the current artifact to an existing one.

        Returns True if the artifact exists, False otherwise.
        """
        if artifact_id in self._artifacts:
            self._current_id = artifact_id
            return True
        if self.load_persisted(artifact_id, make_current=True) is not None:
            return True
        return False

    def list_artifacts(self) -> list[dict[str, Any]]:
        """List all artifacts with their metadata."""
        loaded = [
            {**a.to_dict(), "is_current": a.artifact_id == self._current_id}
            for a in self._artifacts.values()
        ]
        loaded_ids = {item["artifact_id"] for item in loaded}
        for record in self._disk_catalog.values():
            if record.artifact_id in loaded_ids:
                continue
            loaded.append({
                "artifact_id": record.artifact_id,
                "content_length": record.size_bytes,
                "language": record.language,
                "section_count": 0,
                "section_names": [],
                "created_at": record.modified_at,
                "modified_at": record.modified_at,
                "origin": "loaded",
                "turn_id": None,
                "produced_by_model": None,
                "parent_id": None,
                "session_number": record.session_number,
                "version": record.version,
                "descriptor": record.descriptor,
                "source_filename": record.source_filename,
                "saved_path": str(record.path),
                "is_current": record.artifact_id == self._current_id,
            })
        return loaded

    def artifact_candidates(self) -> list[dict[str, Any]]:
        """Return every current-session and persisted version candidate."""
        candidates: list[dict[str, Any]] = []
        for item in self.list_artifacts():
            candidates.append({
                "artifact_id": item["artifact_id"],
                "descriptor": item.get("descriptor", ""),
                "source_filename": item.get("source_filename"),
                "session_number": item.get("session_number", 0),
                "version": item.get("version", 0),
                "is_current": item.get("is_current", False),
            })
        candidates.sort(
            key=lambda item: (
                not item["is_current"],
                -int(item["session_number"]),
                -int(item["version"]),
                str(item["source_filename"] or item["descriptor"]),
            )
        )
        return candidates

    def resolve_by_name(self, prompt: str) -> list[str]:
        """Find artifacts whose descriptor or source filename matches
        words in the user's prompt.

        A match requires at least 2 consecutive words from the prompt
        to appear (case-insensitive, after slug normalization) in the
        artifact's ``descriptor`` or ``source_filename``.  The
        2-consecutive-word threshold prevents false positives on single
        common words like "the", "file", "code".

        When NO artifact matches by name AND the prompt is a
        modify/fix/update-style request AND a current artifact exists,
        the current artifact is returned (the user's "it"/"this" refers
        to the artifact they are currently working on).  This makes
        "the game is flawed... Modify it" target the just-generated
        artifact instead of hunting for a name match from scratch.

        Returns a list of artifact IDs, ordered with the current
        artifact first.  The caller decides what to do with 0, 1, or
        >1 results:
          - 0 matches: no action (normal routing)
          - 1 match: switch to it as the working artifact
          - >1 matches: defer to the clarification side-channel
        """
        if not prompt or (not self._artifacts and not self._disk_catalog):
            return []
        # Tokenize the prompt into lowercase word sequences.  We
        # check all consecutive 2-word pairs against the descriptor
        # and source_filename.  This is deliberately conservative:
        # a bare single word like "kanban" is a topic mention, not
        # a reference to a specific existing artifact.  Two
        # consecutive words ("kanban file", "the kanban",
        # "kanban demo") is a much stronger signal.
        low = prompt.lower()
        words = re.findall(r"[a-zA-Z0-9]+", low)
        if len(words) < 2:
            return []
        pairs: set[str] = set()
        for i in range(len(words) - 1):
            pairs.add(f"{words[i]} {words[i+1]}")

        version_match = re.search(r"\b(?:version|v)\s*[-#]?\s*(\d+)\b", low)
        requested_version = int(version_match.group(1)) if version_match else None
        records: list[tuple[str, str, str, int]] = []
        for art in self._artifacts.values():
            records.append((art.artifact_id, art.descriptor or "", art.source_filename or "", art.version))
        loaded_ids = {item[0] for item in records}
        for record in self._disk_catalog.values():
            if record.artifact_id not in loaded_ids:
                records.append((record.artifact_id, record.descriptor, record.source_filename, record.version))

        exact_filename_matches = [
            artifact_id
            for artifact_id, _descriptor, source_filename, _version in records
            if source_filename and source_filename.lower() in low
        ]
        if exact_filename_matches:
            return exact_filename_matches

        matches: list[str] = []
        for artifact_id, descriptor, source_filename, version in records:
            if requested_version is not None and version != requested_version:
                continue
            descriptor_low = descriptor.lower().replace("-", " ")
            filename_low = pathlib.Path(source_filename).stem.lower().replace("_", " ").replace("-", " ") if source_filename else ""
            # Check 2-word pairs against descriptor and filename.
            for pair in pairs:
                if pair in descriptor_low or pair in filename_low:
                    matches.append(artifact_id)
                    break

        # Current-artifact preference (2026-09-01): when nothing matched
        # by name and the prompt is a modify/fix/update-style request,
        # resolve to the current artifact.  The user's "it"/"this"
        # refers to the artifact they are currently working on (e.g.
        # "the game is flawed... Modify it" after generating the game).
        # This only fires when there are ZERO name matches, so an
        # explicit name reference ("the kanban file") still wins.
        if not matches and self._current_id:
            if re.search(
                r"\b(modify|fix|update|change|edit|correct|improve|rewrite|adjust|repair)\b",
                low,
            ):
                matches.append(self._current_id)

        # Sort: current artifact first, then by creation time
        # (newest first) so the most recently worked-on artifact
        # comes before older ones.
        matches.sort(
            key=lambda aid: (
                aid != self._current_id,
                -(
                    self._artifacts[aid].created_at
                    if aid in self._artifacts
                    else self._disk_catalog[aid].modified_at
                ),
            )
        )
        return matches

    @property
    def current_id(self) -> str | None:
        """The ID of the current artifact."""
        return self._current_id

    @property
    def current(self) -> Artifact | None:
        """The current artifact."""
        return self.get()

    def __len__(self) -> int:
        return len(self._artifacts)

    # ------------------------------------------------------------------
    # Section operations
    # ------------------------------------------------------------------

    def get_section(
        self,
        name: str | None = None,
        *,
        query: str | None = None,
        artifact_id: str | None = None,
    ) -> ArtifactSection | None:
        """Get a section by name or by BM25 search.

        If ``name`` is provided, find the section with that name.
        If ``query`` is provided, search via BM25 for the best match.
        If both are provided, name takes precedence.
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return None

        # By name
        if name:
            norm = _norm_section_name(name)
            for section in artifact.sections:
                if section.name and _norm_section_name(section.name) == norm:
                    return section
            return None

        # By BM25 search
        if query:
            results = self._chunk_index.search(query, k=1)
            if results:
                chunk, score = results[0]
                # Find the matching section
                for section in artifact.sections:
                    if section.chunk and section.chunk.chunk_id == chunk.chunk_id:
                        return section
            return None

        return None

    def get_sections(
        self,
        *,
        query: str | None = None,
        k: int = 3,
        artifact_id: str | None = None,
    ) -> list[ArtifactSection]:
        """Get multiple sections, optionally filtered by BM25 search.

        If ``query`` is None, returns all sections.
        If ``query`` is provided, returns the top-K matching sections.
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return []

        if query is None:
            return list(artifact.sections)

        results = self._chunk_index.search(query, k=k)
        if not results:
            return []

        matched_ids = {chunk.chunk_id for chunk, _ in results}
        return [
            section for section in artifact.sections
            if section.chunk and section.chunk.chunk_id in matched_ids
        ]

    def list_sections(
        self,
        artifact_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all sections with their names and sizes."""
        artifact = self.get(artifact_id)
        if not artifact:
            return []
        return [
            {
                "name": s.name,
                "kind": s.kind,
                "char_start": s.char_start,
                "char_end": s.char_end,
                "length": s.char_end - s.char_start,
            }
            for s in artifact.sections
        ]

    def update_section(
        self,
        name: str,
        new_text: str,
        *,
        artifact_id: str | None = None,
    ) -> bool:
        """Replace a section's content with new text.

        Updates the artifact's content in place and re-parses
        sections.  Returns True if the section was found and
        updated, False otherwise.
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return False

        # Find the section
        section = None
        for s in artifact.sections:
            if s.name == name:
                section = s
                break

        if section is None:
            return False

        # Replace the section's content in the artifact
        old_text = section.text
        new_content = (
            artifact.content[:section.char_start]
            + new_text
            + artifact.content[section.char_end:]
        )
        length_diff = len(new_text) - len(old_text)

        # Update the artifact
        artifact.content = new_content
        artifact.modified_at = time.time()

        # Re-parse sections (the simplest correct approach —
        # re-parse the whole artifact after any modification)
        self._parse_sections(artifact)

        # Re-index in BM25
        self._reindex_artifact(artifact)

        return True

    def add_section(
        self,
        text: str,
        *,
        name: str | None = None,
        kind: str = "code",
        artifact_id: str | None = None,
    ) -> bool:
        """Add a new section to the end of the artifact.

        Returns True if added, False if no artifact exists.
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return False

        # Append to the content
        separator = "\n\n" if artifact.content and not artifact.content.endswith("\n") else ""
        artifact.content = artifact.content + separator + text
        artifact.modified_at = time.time()

        # Re-parse and re-index
        self._parse_sections(artifact)
        self._reindex_artifact(artifact)

        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int = 3,
        artifact_id: str | None = None,
    ) -> list[tuple[ArtifactSection, float]]:
        """Search for sections matching the query via BM25.

        Returns a list of (section, score) pairs, highest first.
        """
        results = self._chunk_index.search(query, k=k)
        if not results:
            return []

        artifact = self.get(artifact_id)
        if not artifact:
            return []

        matched = []
        for chunk, score in results:
            for section in artifact.sections:
                if section.chunk and section.chunk.chunk_id == chunk.chunk_id:
                    matched.append((section, score))
                    break

        return matched

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all artifacts."""
        self._artifacts.clear()
        self._current_id = None
        self._chunk_index.clear()

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        current = self.current
        return {
            "n_artifacts": len(self._artifacts),
            "current_id": self._current_id,
            "current_content_length": len(current.content) if current else 0,
            "current_section_count": len(current.sections) if current else 0,
            "current_sections": self.list_sections() if current else [],
            "chunk_index_size": len(self._chunk_index),
        }

    # ------------------------------------------------------------------
    # Reintegration engine — deterministic splice of model edits
    # ------------------------------------------------------------------

    def apply_edits(
        self,
        edits: dict[str, Any],
        *,
        artifact_id: str | None = None,
        user_intent: str = "",
        source_filename: str | None = None,
        preview: bool = False,
        guard_config: dict[str, Any] | None = None,
        modification_request: str | None = None,
        modification_method: str | None = None,
        modification_model: str | None = None,
    ) -> dict[str, Any]:
        """Compute a structured edit against an artifact and fork a new
        version -- never mutates the source artifact in place.

        This is the reintegration engine.  It takes the JSON output
        from round 2 (replacements + insertions), splices them into
        the artifact content using tree-sitter section boundaries,
        and commits the result as a brand-new ``Artifact`` object.
        The source artifact's ``content``/``sections`` are left
        completely untouched, so it remains independently
        retrievable (branching guarantee).

        All splice operations are applied right-to-left so earlier
        char offsets are not affected by later splices.

        Versioning (Step 5, 2026-08-04):

        * ``artifact.version == 0`` (unpromoted staging entry) ->
          **promote**: assigns a fresh lineage (``session_number``),
          ``version=1``, ``parent_id=None`` (the staging entry is
          discarded, never linked as a parent -- it was never
          surfaced to the user as a version).
        * ``artifact.version >= 1`` -> **fork**: ``version`` is
          ``1 + `` the highest version issued so far in that
          lineage, ``parent_id`` = the source artifact's id,
          ``descriptor``/``session_number``/``source_filename``
          inherited unchanged.
        * **Both cases always mint a brand-new ``artifact_id``** --
          promotion never reuses the staging entry's id.

        The descriptor is a composite of the filename and intent
        slugs when both are available (only for the promotion case;
        forks inherit the descriptor unchanged):

        * both known: ``"modify-<filename-slug>-<intent-slug>"``
        * filename only: ``"modify-<filename-slug>"``
        * intent only (or neither): ``"modify-<intent-slug-or-untitled>"``

        If ``edits`` contains any unresolvable names/anchors, **no**
        new artifact is created -- the source artifact and
        ``current_id`` are left completely unchanged, and the
        returned dict's ``errors`` list is non-empty with no
        ``new_artifact_id`` key.

        Parameters
        ----------
        edits
            The structured edit dict from round 2::

                {
                    "replacements": [
                        {"name": "handleMouse", "new_text": "..."}
                    ],
                    "insertions": [
                        {"anchor_section": "updateWater", "position": "after",
                         "new_text": "function updateFire(r,c) { ... }"}
                    ]
                }
        user_intent
            The router's one-line restatement of user intent, used
            for the descriptor when promoting a staging entry.
        source_filename
            The attachment's original filename stem (if any), used
            for the descriptor when promoting a staging entry.
        preview
            Validate and splice the candidate in memory without creating
            a child artifact, changing ``current_id``, or saving to disk.

        Returns
        -------
        dict
            Summary of applied edits::

                {
                    "sections_replaced": ["handleMouse"],
                    "sections_added": ["updateFire"],
                    "sections_deleted": [],
                    "chars_changed": 423,
                    "artifact_size_before": 23006,
                    "artifact_size_after": 23429,
                    "errors": [],
                    "new_artifact_id": "artifact_15",
                }
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return {
                "sections_replaced": [],
                "sections_added": [],
                "sections_deleted": [],
                "chars_changed": 0,
                "artifact_size_before": 0,
                "artifact_size_after": 0,
                "errors": ["no current artifact"],
            }

        size_before = len(artifact.content)
        errors: list[str] = []
        sections_replaced: list[str] = []
        sections_added: list[str] = []

        # --- Whole-artifact dialect (2026-09-12) -----------------------
        # runtime.modify_artifact_round2_contract's unified-diff fallback
        # verifies a diff with apply_unified_diff's context check and then
        # packages the result as a SINGLE replacement named
        # ``__whole_artifact__`` (dialect "unified_diff").  That sentinel
        # is deliberately skipped by the contract's per-name anchor
        # validation, and the runtime comment asserted "server.py consumes
        # the sentinel identically" -- which was never true: no caller
        # had sentinel handling, so every diff-dialect answer committed
        # through this method failed with
        # "section '__whole_artifact__' not found in artifact".  The
        # dialect was therefore accepted by the contract and ALWAYS
        # rejected at commit (reproduced 2026-09-12).
        #
        # Handle it here, where the fork/commit lives, so the dialect is
        # complete: the sentinel means "replace the entire artifact
        # content".  Correctness was already established by
        # apply_unified_diff (patch parse + verbatim context match) and by
        # the contract's no-change rule, so no section anchoring applies.
        # The shared fork tail below still runs, so versioning, disk
        # lineage continuity, persistence and the diff display are
        # identical to the section-based path (build_diff_response's
        # ``new_section`` lookup simply finds no such section and emits no
        # named hunk, keeping the display honest).
        _whole = None
        for _rep in edits.get("replacements", []):
            if isinstance(_rep, dict) and _rep.get("name") == "__whole_artifact__":
                _whole = _rep
                break
        if _whole is not None:
            _whole_text = _whole.get("new_text")
            if not isinstance(_whole_text, str) or not _whole_text.strip():
                return {
                    "sections_replaced": [],
                    "sections_added": [],
                    "sections_deleted": [],
                    "chars_changed": 0,
                    "artifact_size_before": size_before,
                    "artifact_size_after": size_before,
                    "errors": ["__whole_artifact__ replacement has no content"],
                }
            if len(edits.get("replacements", [])) > 1 or edits.get("insertions") \
                    or edits.get("removals"):
                return {
                    "sections_replaced": [],
                    "sections_added": [],
                    "sections_deleted": [],
                    "chars_changed": 0,
                    "artifact_size_before": size_before,
                    "artifact_size_after": size_before,
                    "errors": [
                        "__whole_artifact__ must be the only edit in the package "
                        "(it replaces the entire artifact)"
                    ],
                }
            content = _whole_text
        else:
            content = None

        # Build a normalized name -> section map for validation.
        # ``_norm_section_name`` strips leading CSS comments and
        # collapses whitespace so the model's edit target matches
        # even when the stored name has a comment prefix or
        # whitespace variation.
        # 2026-09-12: on the whole-artifact path there is no section to
        # resolve (and none should be: the sentinel IS the whole file),
        # so the map is left empty and every per-name loop below skips
        # itself without needing special cases.
        section_map: dict[str, ArtifactSection] = {}
        if content is None:
            for s in artifact.sections:
                if s.name:
                    section_map[_norm_section_name(s.name)] = s

        # Collect all splice operations as (offset, old_len, new_text) tuples.
        # We apply them right-to-left (highest offset first) so earlier
        # offsets remain valid.
        splices: list[tuple[int, int, str]] = []

        # --- Replacements ---
        for rep in ([] if content is not None else edits.get("replacements", [])):
            name = rep.get("name")
            new_text = rep.get("new_text", "")
            if not name:
                errors.append("replacement missing 'name'")
                continue
            section = section_map.get(_norm_section_name(name))
            if not section:
                errors.append(f"section '{name}' not found in artifact")
                continue
            splices.append((section.char_start, section.char_end - section.char_start, new_text))
            sections_replaced.append(name)

        # --- Insertions ---
        for ins in edits.get("insertions", []):
            anchor_name = ins.get("anchor_section")
            position = ins.get("position", "after")
            new_text = ins.get("new_text", "")
            if not anchor_name:
                errors.append("insertion missing 'anchor_section'")
                continue
            anchor = section_map.get(_norm_section_name(anchor_name))
            if not anchor:
                errors.append(f"anchor section '{anchor_name}' not found")
                continue
            if position == "after":
                insert_offset = anchor.char_end
            elif position == "before":
                insert_offset = anchor.char_start
            else:
                errors.append(f"invalid position '{position}' for insertion")
                continue
            # Insertion: old_len=0 at insert_offset
            splices.append((insert_offset, 0, new_text))
            # Try to extract a name for the new section
            new_name = ins.get("name")
            if new_name:
                sections_added.append(new_name)
            else:
                sections_added.append("<unnamed>")

        # --- Removals (2026-09-08) ---
        # A removal is a splice of (char_start, len, ""): the section's
        # text is deleted.  Conflict guards here are the LAST line of
        # defense (the R2 contract checks first) — apply_edits must
        # stay safe when called directly (e.g. the T3 path, or a
        # future caller) without the contract layer.
        removal_names: list[str] = []
        replacement_names: list[str] = []
        for rep in edits.get("replacements", []):
            if isinstance(rep, dict) and rep.get("name"):
                replacement_names.append(_norm_section_name(rep["name"]))
        for removal in edits.get("removals", []) or []:
            if not isinstance(removal, dict):
                errors.append("removals entries must be objects")
                continue
            name = removal.get("name")
            if not name:
                errors.append("removal missing 'name'")
                continue
            section = section_map.get(_norm_section_name(name))
            if not section:
                errors.append(f"removal section '{name}' not found in artifact")
                continue
            if _norm_section_name(name) in replacement_names:
                errors.append(
                    f"section '{name}' is targeted for both replacement and removal"
                )
                continue
            splices.append((section.char_start, section.char_end - section.char_start, ""))
            removal_names.append(name)
        for ins in edits.get("insertions", []):
            if isinstance(ins, dict) and ins.get("anchor_section"):
                anchor_norm = _norm_section_name(ins["anchor_section"])
                if anchor_norm in {_norm_section_name(n) for n in removal_names}:
                    errors.append(
                        f"insertion anchors on removed section '{ins['anchor_section']}'"
                    )

        # Step 5: a functional failure (bad anchor, etc.) must not
        # fork.  The source artifact and _current_id are left
        # completely untouched; no new_artifact_id key is returned.
        if errors:
            return {
                "sections_replaced": [],
                "sections_added": [],
                "sections_deleted": [],
                "chars_changed": 0,
                "artifact_size_before": size_before,
                "artifact_size_after": size_before,
                "errors": errors,
            }

        # Sort splices right-to-left.  2026-09-08 tiebreak: at equal
        # offsets, apply the LARGER old_len first.  This matters when
        # an insertion lands at a section boundary that a removal also
        # starts at (anchor.char_end == removed.char_start): the
        # deletion must consume its span BEFORE the insertion expands
        # the content at the same offset, otherwise the removal
        # splices away the just-inserted text (pre-existing hazard
        # made reachable by removals).
        splices.sort(key=lambda x: (-x[0], -x[1]))

        # Apply splices -- computed against the SOURCE content; the
        # source artifact object itself is never mutated (fork-not-
        # mutate).  For the whole-artifact dialect ``content`` is already
        # the patched text: there are no splices to run (the package is
        # validated single-edit above), so the loop is a no-op there.
        if content is None:
            content = artifact.content
        for offset, old_len, new_text in splices:
            content = content[:offset] + new_text + content[offset + old_len:]

        # 2026-09-09 invented-section signal (user-approved addition):
        # for each REPLACEMENT, compare the new_text against the original
        # section text. A replacement that shares almost no content with
        # the original is a signal the model INVENTED the section instead
        # of modifying it (the 2026-09-09 live failure: render rewritten
        # from imagination while the app still functioned). Computed
        # BEFORE the syntax gate so a syntax-failed candidate ALSO
        # surfaces the invented evidence (an invented section frequently
        # breaks the artifact's language grammar -- that is itself part
        # of the signal).
        # 2026-09-11 (user-ruled upgrade, v7 live failure): the signal
        # becomes a BLOCKING guard when BOTH conditions trip — token
        # overlap below min_token_overlap AND new_text size ratio below
        # min_size_ratio. The div#board gutting (1428 -> 150 chars, 5%
        # overlap) passed the warn-only signal; a stub replacement of a
        # whole section is destruction, not modification. Thresholds are
        # config-owned per tier (local 0.40/0.20, cloud 0.20/0.10; cloud
        # is the ladder's safety net and gets leniency). Scope:
        # REPLACEMENTS ONLY — removals carry name+reason (no new_text,
        # no baseline) and insertions have no original section, so they
        # can never trip this guard. Guard disabled (or absent config ->
        # legacy warn-only behavior at 0.10 overlap).
        warnings: list[str] = []
        guard_errors: list[str] = []
        # 2026-09-11: guard profile resolution. The caller (contract) passes
        # the RESOLVED tier profile (enabled/mins + _tier) via guard_config;
        # absent config -> legacy warn-only behavior.
        guard_cfg = dict(guard_config or {})
        _guard_enabled = bool(guard_cfg.get("enabled", False))
        if _guard_enabled:
            _prof = guard_cfg.get(
                "cloud" if str(guard_cfg.get("_tier", "")).upper().startswith(("T3", "CLOUD")) else "local",
                {},
            )
            _min_ratio = float(_prof.get("min_size_ratio", 0.0) or 0.0)
            _min_overlap = float(_prof.get("min_token_overlap", 0.0) or 0.0)
        for rep in edits.get("replacements", []):
            if not isinstance(rep, dict):
                continue
            name = rep.get("name")
            # 2026-09-12: the whole-artifact dialect carries no section
            # baseline -- apply_unified_diff already verified the patch
            # (context match) and the contract rejected a no-change
            # result.  ``section_map`` is empty on this path, so the
            # lookup below would skip it anyway; the explicit skip keeps
            # that intent obvious rather than incidental.
            if name == "__whole_artifact__":
                continue
            section = section_map.get(_norm_section_name(name or ""))
            if not section:
                continue
            original = section.text
            replacement = rep.get("new_text", "")

            def _toks(s: str) -> set[str]:
                return {t.lower() for t in re.findall(r"\w+", s) if len(t) >= 4}

            orig_tokens = _toks(original)
            if not orig_tokens:
                continue
            new_tokens = _toks(replacement)
            overlap = len(orig_tokens & new_tokens) / len(orig_tokens)
            if _guard_enabled and overlap < _min_overlap:
                orig_len = max(len(original), 1)
                size_ratio = len(replacement) / orig_len
                if size_ratio < _min_ratio:
                    guard_errors.append(
                        f"invented_section_rejected: '{name}' replacement shares only "
                        f"{overlap:.0%} of the original section's tokens (min "
                        f"{_min_overlap:.0%}) and is {size_ratio:.0%} of its size "
                        f"(min {_min_ratio:.0%}) -- the replacement invents rather "
                        f"than modifies the section; rewrite it faithfully or "
                        f"escalate to the cloud tier"
                    )
                    continue
            if overlap < 0.10:
                warnings.append(
                    f"invented_section_signal: '{name}' replacement shares "
                    f"only {overlap:.0%} of the original section's tokens -- "
                    "the model may have invented this section instead of "
                    "modifying it"
                )

        # 2026-09-11 (user-ruled, v8 live failure): SEMANTIC CONFLICT
        # GUARDS, both rejection categories, config-owned in the same
        # guard block. The v8 failure class: FLUENT same-vocabulary
        # invention that the token-overlap guard cannot see (the invented
        # render shared 40% tokens at 1253% size). Two exact signatures:
        #
        #   1. duplicate_declaration (reject): a replacement's new_text
        #      DECLARES a top-level symbol (function/class/let-const-var)
        #      that another section in the same artifact ALSO declares and
        #      that other section is NOT being replaced in this package.
        #      The later declaration silently wins at runtime; the
        #      replaced section was aimed at the wrong target.
        #   2. orphaned_symbol (reject): the replaced section DECLARED
        #      symbol X and the replacement no longer declares X, while
        #      OTHER (non-replaced) sections still REFERENCE X. Every
        #      caller now binds to nothing -- almost always breaking.
        #      Renames must be accompanied by the caller updates.
        #
        # Both are computable from data the section map already holds
        # (name + text + kind); BM25 is a retrieval structure and adds
        # fuzz where this needs exactness. JS/Python declaration grammars
        # only; other languages skip these guards silently.
        semantic_cfg = guard_cfg.get("semantic_conflict_guard") or {}
        if bool(semantic_cfg.get("enabled", False)) and (
            language_js_py := ("javascript" if (artifact.language or "") in ("web", "javascript", "html", "code") else None)
            or ("python" if (artifact.language or "") == "python" else None)
        ):
            _DECL_PATTERNS = {
                # 2026-09-11 (false-positive fix, 'col' case): only
                # function/class declarations count as TOP-LEVEL symbols.
                # const/let/var are excluded — inside function bodies they
                # are locals (the 'col' false positive); statement-position
                # heuristics cannot distinguish them reliably without a
                # full parse. Function/class keyword declarations in these
                # artifact families are always the section-defining symbol.
                "javascript": [
                    re.compile(r"\b(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\("),
                    re.compile(r"\bclass\s+([A-Za-z_$][\w$]*)"),
                ],
                "python": [
                    re.compile(r"^\s*def\s+([A-Za-z_]\w*)", re.M),
                    re.compile(r"^\s*class\s+([A-Za-z_]\w*)", re.M),
                ],
            }
            _decl_res = _DECL_PATTERNS.get(language_js_py, [])
            # Map every section's declared top-level symbols.
            _declared: dict[str, list[tuple[str, str]]] = {}  # symbol -> [(section_name, span)]
            for sec in artifact.sections:
                if not sec.name:
                    continue
                for pat in _decl_res:
                    for m in pat.finditer(sec.text):
                        _declared.setdefault(m.group(1), []).append(
                            (sec.name, f"@ [{sec.char_start}-{sec.char_end}]")
                        )
            # Symbols whose sections are being replaced/removed in THIS
            # package: their declarations may legitimately move/disappear.
            replaced_names = {
                _norm_section_name(r.get("name") or "")
                for r in edits.get("replacements", []) if isinstance(r, dict)
            } | {
                _norm_section_name(r.get("name") or "")
                for r in edits.get("removals", []) if isinstance(r, dict)
            }
            for rep in edits.get("replacements", []):
                if not isinstance(rep, dict):
                    continue
                name = rep.get("name")
                section = section_map.get(_norm_section_name(name or ""))
                if not section:
                    continue
                original = section.text
                replacement = rep.get("new_text", "")
                orig_syms = set()
                for pat in _decl_res:
                    orig_syms |= {m.group(1) for m in pat.finditer(original)}
                new_syms = set()
                for pat in _decl_res:
                    new_syms |= {m.group(1) for m in pat.finditer(replacement)}
                # 1. duplicate declaration: new symbol already declared by
                # a section that this package does NOT touch.
                for sym in new_syms:
                    for other_name, other_span in _declared.get(sym, []):
                        if other_name == section.name:
                            continue
                        if _norm_section_name(other_name) in replaced_names:
                            continue  # that section is being replaced too
                        guard_errors.append(
                            f"duplicate_declaration: '{sym}' is already declared "
                            f"in section '{other_name}' {other_span}, which this "
                            f"package does not replace -- a second declaration "
                            f"shadows it; replace THAT section or declare a "
                            f"different name"
                        )
                        break  # one report per replacement is enough
                # 2. orphaned symbol: the section declared X; the
                # replacement no longer declares X; other untouched
                # sections still REFERENCE X (word-boundary scan).
                for sym in orig_syms - new_syms:
                    if sym in new_syms:
                        continue
                    references = 0
                    ref_sections: list[str] = []
                    word_re = re.compile(rf"\b{re.escape(sym)}\b")
                    for sec in artifact.sections:
                        if sec.name == section.name or _norm_section_name(sec.name or "") in replaced_names:
                            continue
                        if word_re.search(sec.text):
                            references += 1
                            ref_sections.append(sec.name or "?")
                    if references:
                        guard_errors.append(
                            f"orphaned_symbol_rejected: '{sym}' was declared in "
                            f"'{section.name}' but the replacement no longer "
                            f"declares it while {references} other section(s) "
                            f"({', '.join(ref_sections[:3])}) still reference it -- "
                            f"update the callers or keep the declaration"
                        )

        syntax_result = validate_artifact_syntax(content, artifact.language)
        if syntax_result.errors:
            return {
                "sections_replaced": [],
                "sections_added": [],
                "sections_deleted": [],
                "chars_changed": 0,
                "artifact_size_before": size_before,
                "artifact_size_after": size_before,
                "warnings": warnings,
                "errors": guard_errors + [
                    "syntax_validation_failed: " + error
                    for error in syntax_result.errors
                ],
            }
        if preview:
            return {
                "sections_replaced": sections_replaced,
                "sections_added": sections_added,
                "sections_deleted": removal_names,
                "chars_changed": abs(len(content) - size_before),
                "artifact_size_before": size_before,
                "artifact_size_after": len(content),
                "warnings": warnings,
                "errors": guard_errors,
                "preview": True,
            }

        # --- Step 5: fork-not-mutate, always mint a new id ---
        if artifact.version == 0:
            # Promote the staging entry.  The staging entry itself is
            # discarded (never surfaced to the user as a version) --
            # parent_id is None, not the staging id.
            eff_filename = source_filename or artifact.source_filename
            if eff_filename:
                # Strip the extension -- the descriptor names the
                # lineage's topic, not its file type.
                filename_stem = pathlib.Path(eff_filename).stem
                filename_slug = _slugify(filename_stem)
                intent_slug = _slugify(user_intent) if user_intent else ""
                new_descriptor = (
                    f"modify-{filename_slug}-{intent_slug}" if intent_slug
                    else f"modify-{filename_slug}"
                )
            else:
                new_descriptor = "modify-" + _slugify(user_intent or "untitled")
            # --- Disk-lineage continuity (2026-08-14) ---
            # Before minting a new session_number, check whether a
            # lineage with this descriptor already exists on disk.
            # If so, reuse its session_number and continue the version
            # from max_version + 1.  This makes version increments
            # survive server restarts: loading kanban_demo.html and
            # modifying it always continues the same lineage, not a
            # fresh v1 every time.
            existing = self._disk_lineages.get(new_descriptor)
            if existing is not None:
                new_session_number, max_version = existing
                new_version = max_version + 1
                self._lineage_version_counter[new_session_number] = new_version
                self._disk_lineages[new_descriptor] = (new_session_number, new_version)
            else:
                self._session_counter += 1
                new_session_number = self._session_counter
                new_version = 1
                self._lineage_version_counter[new_session_number] = 1
                self._disk_lineages[new_descriptor] = (new_session_number, 1)
            new_parent_id = None
        else:
            new_session_number = artifact.session_number
            new_version = artifact.version + 1
            self._lineage_version_counter[new_session_number] = max(
                self._lineage_version_counter.get(new_session_number, 0),
                new_version,
            )
            self._disk_lineages[artifact.descriptor] = (
                new_session_number,
                max(self._disk_lineages.get(artifact.descriptor, (0, 0))[1], new_version),
            )
            new_parent_id = artifact.artifact_id
            new_descriptor = artifact.descriptor  # inherited unchanged (5.4.3)

        self._id_counter += 1
        new_artifact_id = f"artifact_{self._id_counter}"  # ALWAYS a fresh id

        # --- Provenance inheritance (spec 5.7.3, added 2026-09-14) ---
        # origin_prompt is INHERITED UNCHANGED -- never rewritten to the
        # combined prompt.  A daughter that stored the combined prompt would
        # silently DROP every earlier modification on the next regeneration,
        # because the combo is rebuilt as origin + log + current.  The log is
        # append-only, so provenance grows monotonically across cloud and
        # local hops in any order and the lineage never loses its basis.
        new_origin_prompt = artifact.origin_prompt
        new_modifications = list(artifact.modifications or [])
        _req_text = (modification_request or "").strip()
        if _req_text:
            try:
                from .provenance import record_modification as _rm
                new_modifications = _rm(
                    new_modifications, _req_text, modification_method or "",
                    modification_model or "",
                )
            except ImportError:
                # provenance.py absent -> keep the inherited log only.
                pass

        new_artifact = Artifact(
            artifact_id=new_artifact_id,
            content=content,
            language=artifact.language,
            origin="forked" if artifact.version >= 1 else artifact.origin,
            turn_id=artifact.turn_id,
            produced_by_model=artifact.produced_by_model,
            produced_by_domain=artifact.produced_by_domain,
            parent_id=new_parent_id,
            session_number=new_session_number,
            version=new_version,
            descriptor=new_descriptor,
            source_filename=source_filename or artifact.source_filename,
            origin_prompt=new_origin_prompt,
            modifications=new_modifications,
        )
        self._parse_sections(new_artifact)
        self._artifacts[new_artifact_id] = new_artifact
        self._index_artifact(new_artifact)
        self._current_id = new_artifact_id
        save_artifact_to_disk(new_artifact)

        size_after = len(new_artifact.content)
        chars_changed = abs(size_after - size_before)

        return {
            "sections_replaced": sections_replaced,
            "sections_added": sections_added,
            "sections_deleted": removal_names,
            "chars_changed": chars_changed,
            "artifact_size_before": size_before,
            "artifact_size_after": size_after,
            "errors": errors,
            "new_artifact_id": new_artifact_id,
        }

    # ------------------------------------------------------------------
    # Context building for round 1 / evaluate
    # ------------------------------------------------------------------

    def build_outline(self, artifact_id: str | None = None) -> str:
        """Build a compact tree-sitter outline of the artifact.

        Returns a text rendering of the artifact's sections with
        their byte ranges and names.  This is passed to the
        primary model in round 1 (canonical truth extraction)
        and in evaluate_artifact so the model can see the
        structure without reading every line.

        Format::

            function handleMouse @ [4521-4893] (806 chars)
            function pourMaterial @ [4895-5210] (315 chars)
            function applyGravity @ [5212-5800] (588 chars)
            ...
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return ""
        lines: list[str] = []
        for section in artifact.sections:
            name = section.name or "<anonymous>"
            kind = section.kind
            start = section.char_start
            end = section.char_end
            length = end - start
            lines.append(f"  {kind} {name} @ [{start}-{end}] ({length} chars)")
        return "\n".join(lines)

    def build_section_index_text(self, artifact_id: str | None = None) -> str:
        """Build a compact BM25 section list for the model.

        Returns a text rendering of each section's top tokens
        (from the BM25 index) so the model can do lexical
        matching when deciding which sections to extract.

        Format::

            handleMouse: mouse, event, clientX, rect, deposit
            pourMaterial: material, grid, deposit, nozzle
            ...
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return ""
        lines: list[str] = []
        for section in artifact.sections:
            name = section.name or "<anonymous>"
            # Extract top tokens from the section text
            from .chunk_index import tokenize
            tokens = tokenize(section.text)
            # Get top 8 unique tokens by frequency
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            top = sorted(counts.items(), key=lambda x: -x[1])[:8]
            top_str = ", ".join(t for t, _ in top)
            lines.append(f"  {name}: {top_str}")
        return "\n".join(lines)

    def build_context_package(
        self,
        *,
        include_content: bool = True,
        include_section_index: bool = True,
        artifact_id: str | None = None,
    ) -> str:
        """Build the full context package for round 1 / evaluate.

        This assembles:
        1. The full artifact content (if ``include_content``)
        2. The tree-sitter outline
        3. The BM25 section index (if ``include_section_index``)

        The result is a single text block that gets appended
        to the system or user message of the primary model call.

        2026-09-18 (user ruling): ``include_section_index`` defaults to True
        so every existing caller is byte-identical, but the CALLERS that feed
        a model prompt must pass False unless the harness's own cloud prompt
        actually promises a section index.  BM25 was de-legacized from the
        cloud payload on 2026-09-08, but this builder kept emitting it, and
        ``_build_t3_context`` is what the runtime reads (``t3_context_text``),
        so the de-legacization never took effect for T3.  The only harness
        whose prompt promises a section index is ``evaluate_artifact``.
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return ""
        parts: list[str] = []
        if include_content:
            parts.append("[ARTIFACT CONTENT]")
            parts.append(artifact.content)
            parts.append("")
        outline = self.build_outline(artifact_id)
        if outline:
            parts.append("[ARTIFACT STRUCTURE — tree-sitter outline]")
            parts.append(outline)
            parts.append("")
        if include_section_index:
            section_index = self.build_section_index_text(artifact_id)
            if section_index:
                parts.append("[ARTIFACT SECTIONS — BM25 index]")
                parts.append(section_index)
                parts.append("")
        return "\n".join(parts)

    def build_focused_context(
        self,
        query: str,
        *,
        artifact_id: str | None = None,
        max_sections: int = 8,
        neighbor_radius: int = 1,
        max_chars: int = 12000,
    ) -> str:
        """Build a bounded, authoritative context from matching sections.

        BM25 ranks sections for ``query`` within the selected artifact.
        The returned content is always taken from the artifact store;
        the model never becomes the source of truth for section text.
        Optional neighboring sections provide a small structural closure
        around the lexical matches without sending the whole artifact.
        """
        artifact = self.get(artifact_id)
        if not artifact or not query.strip():
            return ""
        max_sections = max(1, int(max_sections))
        neighbor_radius = max(0, int(neighbor_radius))
        max_chars = max(1000, int(max_chars))
        section_by_chunk_id = {
            section.chunk.chunk_id: section
            for section in artifact.sections
            if section.chunk is not None
        }
        ranked: list[tuple[ArtifactSection, float]] = []
        seen: set[str] = set()
        for chunk, score in self._chunk_index.search_by_turn(
            query, artifact.artifact_id, k=max_sections * 4,
        ):
            section = section_by_chunk_id.get(chunk.chunk_id)
            if section is not None and section.name not in seen:
                ranked.append((section, score))
                seen.add(section.name or f"<anonymous:{section.char_start}>")
        if not ranked:
            return "[FOCUSED ARTIFACT CONTEXT]\nNo matching sections found.\n[END FOCUSED ARTIFACT CONTEXT]"

        selected: dict[int, float] = {
            id(section): score for section, score in ranked[:max_sections]
        }
        if neighbor_radius:
            section_positions = {id(section): i for i, section in enumerate(artifact.sections)}
            for section in list(selected):
                center = section_positions[id(next(s for s in artifact.sections if id(s) == section))]
                for offset in range(1, neighbor_radius + 1):
                    for position in (center - offset, center + offset):
                        if 0 <= position < len(artifact.sections):
                            neighbor = artifact.sections[position]
                            if neighbor.name and len(selected) < max_sections:
                                selected.setdefault(id(neighbor), 0.0)

        ordered = sorted(
            (section for section in artifact.sections if id(section) in selected),
            key=lambda section: section.char_start,
        )
        parts = [
            "[FOCUSED ARTIFACT CONTEXT]",
            f"artifact_id={artifact.artifact_id} language={artifact.language or ''}",
            "The following section contents are authoritative and may be edited only when justified by the user request.",
        ]
        used_chars = 0
        for section in ordered:
            content = section.text or ""
            if used_chars + len(content) > max_chars:
                continue
            score = selected[id(section)]
            parts.extend([
                "",
                f"[SECTION name={section.name or '<anonymous>'} kind={section.kind} "
                f"char_start={section.char_start} char_end={section.char_end} "
                f"retrieval_score={score:.4f}]",
                content,
                "[END SECTION]",
            ])
            used_chars += len(content)
        parts.append("[END FOCUSED ARTIFACT CONTEXT]")
        return "\n".join(parts)

    def build_named_context(
        self,
        names: list[str],
        *,
        artifact_id: str | None = None,
        max_chars: int = 12000,
    ) -> str:
        """Build bounded authoritative context for named sections."""
        artifact = self.get(artifact_id)
        if not artifact or not names:
            return ""
        wanted = {_norm_section_name(name) for name in names if name}
        selected = [
            section for section in artifact.sections
            if section.name and _norm_section_name(section.name) in wanted
        ]
        parts = [
            "[AUTHORITATIVE SELECTED ARTIFACT SECTIONS]",
            f"artifact_id={artifact.artifact_id} language={artifact.language or ''}",
        ]
        used_chars = 0
        for section in selected:
            content = section.text or ""
            if used_chars + len(content) > max_chars:
                continue
            parts.extend([
                "",
                f"[SECTION name={section.name} kind={section.kind} "
                f"char_start={section.char_start} char_end={section.char_end}]",
                content,
                "[END SECTION]",
            ])
            used_chars += len(content)
        parts.append("[END AUTHORITATIVE SELECTED ARTIFACT SECTIONS]")
        return "\n".join(parts)

    def get_section_names(self, artifact_id: str | None = None) -> list[str]:
        """Return the list of section names for the current artifact.

        Used by the router to pass artifact section names as hints
        so the router model can decide if the prompt is a modify/
        evaluate of the existing artifact or a new task.
        """
        artifact = self.get(artifact_id)
        if not artifact:
            return []
        return [s.name for s in artifact.sections if s.name]

    # ------------------------------------------------------------------
    # Internal: section parsing and indexing
    # ------------------------------------------------------------------

    def _parse_sections(self, artifact: Artifact) -> None:
        """Parse the artifact content into sections using the chunker."""
        chunks = chunk_model_output(
            artifact.content,
            turn_id=artifact.artifact_id,
            language=artifact.language,
        )
        content_bytes = artifact.content.encode("utf-8")

        def byte_to_char(byte_offset: int) -> int:
            return len(content_bytes[:byte_offset].decode("utf-8", errors="replace"))

        sections: list[ArtifactSection] = []
        for chunk in chunks:
            char_start = byte_to_char(chunk.char_start)
            char_end = byte_to_char(chunk.char_end)
            name = chunk.symbol or _synthetic_section_name(
                chunk.kind, char_start, char_end,
            )
            sections.append(ArtifactSection(
                name=name,
                kind=chunk.kind,
                text=chunk.text,
                char_start=char_start,
                char_end=char_end,
                chunk=chunk,
            ))
        artifact.sections = sections

    def _index_artifact(self, artifact: Artifact) -> None:
        """Index an artifact's sections in the BM25 index."""
        for section in artifact.sections:
            if section.chunk:
                self._chunk_index.insert(section.chunk)

    def _reindex_artifact(self, artifact: Artifact) -> None:
        """Remove old chunks and re-index an artifact after modification."""
        # Remove old chunks for this artifact
        self._chunk_index.remove_turn(artifact.artifact_id)
        # Re-index with new sections
        self._index_artifact(artifact)

    def _evict_if_needed(self) -> None:
        """Evict the oldest artifact if over the limit."""
        while len(self._artifacts) > self._max_artifacts:
            # Find the oldest artifact by created_at
            oldest_id = min(
                self._artifacts,
                key=lambda aid: self._artifacts[aid].created_at,
            )
            # Remove from BM25 index
            self._chunk_index.remove_turn(oldest_id)
            # Remove from store
            del self._artifacts[oldest_id]
            # If we evicted the current artifact, pick a new one
            if self._current_id == oldest_id:
                self._current_id = next(
                    iter(self._artifacts), None
                )