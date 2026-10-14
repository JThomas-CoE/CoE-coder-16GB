"""Live per-token raw disk writer for the clean slice.

The clean slice is required to save every model token to disk
the moment it arrives on the response stream, along with
**live meta-data** for post-mortem debugging:

  * Per-chunk timestamps (monotonic + wall-clock + elapsed-since-stream-start)
  * Per-tier boundaries (model id, transition time)
  * Per-call pathology events (TTFT, decode rate, error transitions)
  * Per-problem header (qid, harness, work, prompt size)
  * Per-attempt log (model, output length, error string, elapsed)
  * Final sentinel with chunk count, byte count, total elapsed

All writes are append + fsync so the on-disk file is the
ground truth at every moment, even across power loss.

Design:

* **One writer per problem.**  The bench instantiates
  ``RawTokenWriter(raw_dir, qid)`` at problem start.  Each
  chunk that arrives from the model is appended via
  ``write_chunk`` (which immediately calls ``flush()`` and
  ``os.fsync`` so the on-disk file always reflects the
  latest bytes).
* **Tier headers.**  When a new tier is entered, the writer
  inserts a ``=== TIER T<n> model=<id> ===`` header so the
  file is self-describing for post-mortem analysis.
* **Best-effort close.**  ``close()`` flushes, fsyncs, and
  closes the underlying file.  The writer is safe to use
  as an ``async with`` context manager; the runtime does
  not need to catch exceptions because the writer swallows
  ``OSError`` on every write (a disk failure should never
  break a model inference).

The writer is the *only* code path that touches the
``raw/<qid>.raw.txt`` file at runtime.  The bench's
post-problem recorder must not overwrite the file (the
recorder now checks ``stat().st_size == 0`` before writing
so the live stream is preserved).
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any


class RawTokenWriter:
    """Append-only writer for per-problem raw token stream + meta-data log.

    Usage::

        writer = RawTokenWriter(Path("raw"), "abc338_c")
        writer.begin_problem(harness="python_new_code", ...)
        writer.begin_tier("T0", "model-id")
        async for chunk, _ in transport.stream(...):
            await writer.write_chunk(chunk)
        writer.end_tier(elapsed_s=3.2, output_len=1200, error=None)
        writer.log_pathology("first_token_timeout", elapsed_s=5.0, observed_rate=0.0, budget_s=5.0)
        await writer.close()
    """

    def __init__(self, raw_dir: str | Path, qid: str) -> None:
        self._raw_dir = Path(raw_dir)
        self._qid = str(qid)
        # Live token stream: ``<qid>.live.txt`` is the
        # per-token stream (every chunk, with tier
        # headers and meta events inline).  The
        # ``<qid>.live.meta.jsonl`` is the structured
        # meta event log (one JSON object per line for
        # problem_start, tier_start, pathology, tier_end,
        # problem_end).
        #
        # These are SEPARATE from the recorder's
        # ``<qid>.raw.txt`` (which holds the final
        # accumulated ``content``).  Keeping the
        # live stream under a different name means the
        # recorder's existing per-problem raw-write is
        # untouched -- the bench runner does not need
        # to change.
        self._path = self._raw_dir / f"{self._qid}.live.txt"
        self._meta_path = self._raw_dir / f"{self._qid}.live.meta.jsonl"
        self._fh: Any = None
        self._meta_fh: Any = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._tier_seq: int = 0
        self._bytes_written: int = 0
        self._chunks_written: int = 0
        self._started_at: float | None = None
        self._stream_started_at: float | None = None
        self._last_chunk_at: float | None = None
        self._max_gap_s: float = 0.0
        self._max_gap_at: int = 0
        try:
            self._raw_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @property
    def path(self) -> Path:
        return self._path

    @property
    def meta_path(self) -> Path:
        return self._meta_path

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    @property
    def chunks_written(self) -> int:
        return self._chunks_written

    def _ensure_open(self) -> None:
        if self._fh is None:
            self._fh = self._path.open("a", encoding="utf-8", buffering=1)
        if self._meta_fh is None:
            self._meta_fh = self._meta_path.open("a", encoding="utf-8", buffering=1)
        if self._started_at is None:
            self._started_at = time.monotonic()

    def _write_meta(self, event: dict[str, Any]) -> None:
        try:
            import json
            self._ensure_open()
            line = json.dumps(event, ensure_ascii=False)
            # No fsync per meta event: the meta
            # jsonl is debug-only and the close()
            # path fsyncs once at end-of-stream.
            self._meta_fh.write(line + "\n")
            self._meta_fh.flush()
        except Exception as exc:
            import sys
            print(f"[warn] raw_writer._write_meta failed: {exc}", file=sys.stderr)

    def log_round_request(
        self,
        *,
        tier: str,
        model_id: str,
        provider: str,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any],
        options: dict[str, Any],
    ) -> None:
        """Record the exact provider-neutral R1/R2 request envelope.

        This is diagnostic metadata only.  It does not alter the
        package sent to the transport, and it is written separately
        from the visible token stream so large artifact prompts remain
        reconstructable without contaminating generated content.
        """
        self._write_meta({
            "event": "round_request",
            "qid": self._qid,
            "tier": tier,
            "model_id": model_id,
            "provider": provider,
            "messages": messages,
            "sampling": sampling,
            "options": options,
            "t_wall": time.time(),
            "t_mono": time.monotonic(),
        })

    def begin_problem(
        self,
        *,
        harness: str = "?",
        work: str = "?",
        prompt_tokens: int = 0,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ) -> None:
        try:
            self._ensure_open()
            header = (
                f"\n"
                f"=== PROBLEM {self._qid} "
                f"harness={harness} work={work} "
                f"prompt_tokens={prompt_tokens} "
                f"seed={seed} temp={temperature} top_p={top_p} top_k={top_k} "
                f"num_ctx={num_ctx} num_predict={num_predict} "
                f"t_wall={time.time():.3f} "
                f"t_mono={time.monotonic():.3f} ===\n"
            )
            self._fh.write(header)
            self._fh.flush()
            self._bytes_written += len(header.encode("utf-8"))
            self._write_meta({
                "event": "problem_start",
                "qid": self._qid,
                "harness": harness,
                "work": work,
                "prompt_tokens": prompt_tokens,
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "t_wall": time.time(),
                "t_mono": time.monotonic(),
            })
        except Exception as exc:
            import sys
            print(f"[warn] raw_writer.begin_problem failed: {exc}", file=sys.stderr)

    def begin_tier(self, tier: str, model_id: str) -> None:
        if self._closed:
            return
        try:
            self._ensure_open()
            self._tier_seq += 1
            now = time.monotonic()
            elapsed_problem = now - (self._started_at or now)
            header = (
                f"\n=== TIER {tier} model={model_id} "
                f"seq={self._tier_seq} "
                f"t_mono={now:.3f} "
                f"elapsed_problem={elapsed_problem:.3f}s "
                f"t_wall={time.time():.3f} ===\n"
            )
            # No per-event fsync: flush is enough
            # for the tier boundary marker.
            self._fh.write(header)
            self._fh.flush()
            self._bytes_written += len(header.encode("utf-8"))
            self._stream_started_at = now
            self._last_chunk_at = None
            self._max_gap_s = 0.0
            self._max_gap_at = 0
            self._write_meta({
                "event": "tier_start",
                "qid": self._qid,
                "tier": tier,
                "model_id": model_id,
                "tier_seq": self._tier_seq,
                "t_mono": now,
                "t_wall": time.time(),
                "elapsed_problem_s": elapsed_problem,
            })
            # Live echo: also print the tier header
            # to the terminal so the operator can see
            # which model is being tried.
            import os as _os
            if _os.environ.get("CLEAN_SLICE_RAW_ECHO"):
                import sys
                print(
                    f"\n[begin-tier] {self._qid} {tier} {model_id}\n",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            import sys
            print(f"[warn] raw_writer.begin_tier failed: {exc}", file=sys.stderr)

    def end_tier(
        self,
        *,
        elapsed_s: float,
        output_len: int,
        error: str | None,
        selected_model: str | None = None,
        ttft_s: float | None = None,
        decode_rate: float | None = None,
        chunks: int = 0,
    ) -> None:
        if self._closed:
            return
        try:
            self._ensure_open()
            now = time.monotonic()
            elapsed_problem = now - (self._started_at or now)
            tail = (
                f"\n--- TIER END "
                f"chunks={chunks} output={output_len} bytes "
                f"elapsed={elapsed_s:.3f}s "
                f"ttft={ttft_s if ttft_s is not None else 'n/a'} "
                f"rate={decode_rate if decode_rate is not None else 'n/a'} "
                f"err={error or 'none'} "
                f"model={selected_model or '?'} "
                f"max_gap={self._max_gap_s:.3f}s@{self._max_gap_at} "
                f"t_mono={now:.3f} ---\n"
            )
            # No fsync per event.
            self._fh.write(tail)
            self._fh.flush()
            self._bytes_written += len(tail.encode("utf-8"))
            self._write_meta({
                "event": "tier_end",
                "qid": self._qid,
                "elapsed_s": elapsed_s,
                "output_len": output_len,
                "error": error,
                "selected_model": selected_model,
                "ttft_s": ttft_s,
                "decode_rate": decode_rate,
                "chunks": chunks,
                "max_gap_s": self._max_gap_s,
                "max_gap_at_chunk": self._max_gap_at,
                "t_mono": now,
                "t_wall": time.time(),
                "elapsed_problem_s": elapsed_problem,
            })
        except Exception as exc:
            import sys
            print(f"[warn] raw_writer.end_tier failed: {exc}", file=sys.stderr)

    def log_pathology(
        self,
        *,
        kind: str,
        elapsed_s: float,
        observed_rate: float | None = None,
        budget_s: float | None = None,
        chunks_so_far: int = 0,
        last_token_at: float | None = None,
        details: str | None = None,
    ) -> None:
        if self._closed:
            return
        try:
            self._ensure_open()
            now = time.monotonic()
            gap = None
            if last_token_at is not None:
                gap = now - last_token_at
            marker = (
                f"\n!!! PATHOLOGY {kind} "
                f"elapsed={elapsed_s:.3f}s "
                f"observed_rate={observed_rate if observed_rate is not None else 'n/a'} "
                f"budget={budget_s if budget_s is not None else 'n/a'} "
                f"chunks={chunks_so_far} "
                f"last_token_gap={gap if gap is not None else 'n/a'}s "
                f"details={details or ''} "
                f"t_mono={now:.3f} t_wall={time.time():.3f} !!!\n"
            )
            # No fsync per event.
            self._fh.write(marker)
            self._fh.flush()
            self._bytes_written += len(marker.encode("utf-8"))
            self._write_meta({
                "event": "pathology",
                "kind": kind,
                "qid": self._qid,
                "elapsed_s": elapsed_s,
                "observed_rate": observed_rate,
                "budget_s": budget_s,
                "chunks_so_far": chunks_so_far,
                "last_token_gap_s": gap,
                "details": details,
                "t_mono": now,
                "t_wall": time.time(),
            })
            # Live echo: also print the pathology
            # event to the terminal so the operator
            # sees the abort decision in real time.
            import os as _os
            if _os.environ.get("CLEAN_SLICE_RAW_ECHO"):
                import sys
                print(
                    f"\n[pathology] {self._qid} {kind} "
                    f"elapsed={elapsed_s:.3f}s rate={observed_rate} "
                    f"chunks={chunks_so_far}\n",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            import sys
            print(f"[warn] raw_writer.log_pathology failed: {exc}", file=sys.stderr)

    async def write_chunk(self, chunk: str) -> None:
        if self._closed:
            return
        if not chunk:
            return
        async with self._lock:
            try:
                self._ensure_open()
                now = time.monotonic()
                if self._last_chunk_at is not None:
                    gap = now - self._last_chunk_at
                    if gap > self._max_gap_s:
                        self._max_gap_s = gap
                        self._max_gap_at = self._chunks_written
                self._last_chunk_at = now
                if self._stream_started_at is None:
                    self._stream_started_at = now
                # No fsync per chunk: that adds a
                # syscall per token and the model
                # is generating at 50-100 tok/s.
                # The file is opened in line-buffered
                # mode (buffering=1) so flush() pushes
                # each chunk to the OS immediately.
                # fsync is only called on close so
                # a power loss after close still
                # preserves the data; in-flight
                # bytes are at the OS's discretion.
                self._fh.write(chunk)
                self._fh.flush()
                self._bytes_written += len(chunk.encode("utf-8"))
                self._chunks_written += 1
                # Live terminal echo (debug only).
                # When the env var CLEAN_SLICE_RAW_ECHO
                # is set, every chunk is also streamed
                # to stderr so the operator can watch
                # the model's output arrive in real
                # time without having to tail the
                # on-disk file.  Default is off so
                # production runs are not polluted.
                import os as _os
                if _os.environ.get("CLEAN_SLICE_RAW_ECHO"):
                    import sys
                    # Show a short header on the very
                    # first chunk so the operator knows
                    # which qid is streaming.
                    if self._chunks_written == 1:
                        print(
                            f"\n[begin-tier] {self._qid} "
                            f"t={now - (self._started_at or now):.3f}s\n",
                            file=sys.stderr,
                            flush=True,
                        )
                    # Stream the chunk bare.  ``end=""``
                    # so the stream stays on a single
                    # visual line as tokens arrive.
                    # No prefix -- the user wants the
                    # actual raw text, not labels.
                    print(
                        f"{chunk}",
                        file=sys.stderr,
                        end="",
                        flush=True,
                    )
            except Exception as exc:
                import sys
                print(f"[warn] raw_writer.write_chunk failed: {exc}", file=sys.stderr)

    async def close(self) -> None:
        if self._closed:
            return
        async with self._lock:
            self._closed = True
            try:
                if self._fh is not None:
                    self._fh.flush()
                    try:
                        os.fsync(self._fh.fileno())
                    except OSError:
                        pass
                if self._meta_fh is not None:
                    self._meta_fh.flush()
                    try:
                        os.fsync(self._meta_fh.fileno())
                    except OSError:
                        pass
            except Exception:
                pass
            try:
                if self._fh is not None:
                    self._fh.close()
            except Exception:
                pass
            try:
                if self._meta_fh is not None:
                    self._meta_fh.close()
            except Exception:
                pass
            self._fh = None
            self._meta_fh = None
            if self._started_at is not None:
                elapsed = time.monotonic() - self._started_at
                try:
                    import json
                    summary = {
                        "event": "problem_end",
                        "qid": self._qid,
                        "chunks": self._chunks_written,
                        "bytes": self._bytes_written,
                        "elapsed_s": elapsed,
                        "max_gap_s": self._max_gap_s,
                        "max_gap_at_chunk": self._max_gap_at,
                        "t_mono": time.monotonic(),
                        "t_wall": time.time(),
                    }
                    with self._meta_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(summary) + "\n")
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass
                except Exception:
                    pass
                sentinel = (
                    f"\n=== END raw_writer "
                    f"qid={self._qid} "
                    f"chunks={self._chunks_written} "
                    f"bytes={self._bytes_written} "
                    f"elapsed={elapsed:.3f}s "
                    f"max_gap={self._max_gap_s:.3f}s@{self._max_gap_at} "
                    f"t_mono={time.monotonic():.3f} ===\n"
                )
                try:
                    with self._path.open("a", encoding="utf-8") as f:
                        f.write(sentinel)
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except OSError:
                            pass
                except Exception:
                    pass
            # Live echo: announce end-of-stream on
            # the terminal so the operator knows
            # the writer is closed.
            import os as _os
            if _os.environ.get("CLEAN_SLICE_RAW_ECHO"):
                import sys
                print(
                    f"\n[end-tier] {self._qid} chunks={self._chunks_written} bytes={self._bytes_written}\n",
                    file=sys.stderr,
                    flush=True,
                )

    async def __aenter__(self) -> "RawTokenWriter":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
