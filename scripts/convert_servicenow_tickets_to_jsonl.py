#!/usr/bin/env python3
"""
Convert ServiceNow tickets stored as "one JSON object per file" into sharded JSONL.

Why: Ray Data's `read_json` is typically optimized for JSONL and tends to scale better
than reading many small files via `read_binary_files`.

This converter:
- Recursively discovers `*.json` files under an input directory
- Parses each file as a single JSON object
- Optionally drops the top-level `attachments` field
- Writes sharded JSONL (optionally gzip-compressed)
- Adds a `source_path` field for provenance (relative to the input dir by default)
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable


LOG = logging.getLogger("convert_servicenow_tickets_to_jsonl")


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _iter_json_files(input_dir: Path, *, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from input_dir.rglob("*.json")
    else:
        yield from input_dir.glob("*.json")


def _open_text_writer(path: Path, *, gzip_output: bool):
    if gzip_output:
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ServiceNow per-file JSON tickets into sharded JSONL."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-ticket JSON files (one JSON object per file).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write JSONL shard files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse under --input-dir.",
    )
    parser.add_argument(
        "--records-per-shard",
        type=int,
        default=10_000,
        help="Maximum number of tickets per output shard file.",
    )
    parser.add_argument(
        "--output-prefix",
        default="tickets",
        help="Prefix for output shard filenames.",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Write gzip-compressed shards (*.jsonl.gz).",
    )
    parser.add_argument(
        "--keep-attachments",
        action="store_true",
        help="Keep the top-level 'attachments' field (default: drop it).",
    )
    parser.add_argument(
        "--source-path",
        choices=("relative", "absolute"),
        default="relative",
        help="How to store `source_path` in each JSONL record.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="For debugging: only convert the first N discovered JSON files.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=1.0,
        help="Minimum seconds between progress updates (default: 1.0).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _to_source_path(
    p: Path, *, input_dir: Path, mode: str
) -> str:
    if mode == "absolute":
        return str(p)
    try:
        return str(p.relative_to(input_dir))
    except Exception:
        return str(p)


def _next_shard_path(
    output_dir: Path,
    *,
    prefix: str,
    shard_idx: int,
    gzip_output: bool,
) -> Path:
    suffix = ".jsonl.gz" if gzip_output else ".jsonl"
    return output_dir / f"{prefix}-{shard_idx:05d}{suffix}"


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "--:--"
    seconds = max(0.0, seconds)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class _Progress:
    def __init__(
        self,
        *,
        total_files: int,
        enabled: bool,
        interval_s: float,
    ) -> None:
        self._total_files = total_files
        self._enabled = enabled
        self._interval_s = max(0.1, interval_s)
        self._start = time.monotonic()
        self._last = 0.0
        self._isatty = sys.stderr.isatty()

    def update(
        self,
        *,
        processed_files: int,
        written_records: int,
        parse_errors: int,
        shard_idx: int,
    ) -> None:
        if not self._enabled:
            return

        now = time.monotonic()
        if (now - self._last) < self._interval_s and processed_files < self._total_files:
            return
        self._last = now

        elapsed = max(1e-6, now - self._start)
        rate = processed_files / elapsed
        remaining = self._total_files - processed_files
        eta_s = (remaining / rate) if rate > 0 else None
        pct = (processed_files / self._total_files * 100.0) if self._total_files else 100.0

        if self._isatty:
            bar_len = 28
            filled = (
                int(processed_files / self._total_files * bar_len)
                if self._total_files
                else bar_len
            )
            bar = "#" * filled + "-" * (bar_len - filled)
            msg = (
                f"\r[{bar}] {pct:5.1f}%  "
                f"{processed_files}/{self._total_files}  "
                f"written={written_records}  "
                f"errors={parse_errors}  "
                f"shards={shard_idx + 1}  "
                f"{rate:,.0f} files/s  "
                f"ETA {_format_eta(eta_s)}"
            )
            sys.stderr.write(msg)
            sys.stderr.flush()
        else:
            LOG.info(
                "Progress: %d/%d (%.1f%%) written=%d errors=%d shards=%d rate=%.0f files/s ETA=%s",
                processed_files,
                self._total_files,
                pct,
                written_records,
                parse_errors,
                shard_idx + 1,
                rate,
                _format_eta(eta_s),
            )

    def finish(self) -> None:
        if self._enabled and self._isatty:
            sys.stderr.write("\n")
            sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir is not a directory: {input_dir}")

    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir, overwrite=bool(args.overwrite))

    recursive = not args.no_recursive
    files = sorted(p for p in _iter_json_files(input_dir, recursive=recursive) if p.is_file())
    if args.limit_files is not None:
        files = files[: args.limit_files]

    if not files:
        raise SystemExit(f"No *.json files found under: {input_dir}")

    LOG.info("Discovered %d JSON file(s).", len(files))

    shard_idx = 0
    shard_records = 0
    written = 0
    parse_errors = 0

    progress = _Progress(
        total_files=len(files),
        enabled=not bool(args.no_progress),
        interval_s=float(args.progress_interval_seconds),
    )

    shard_path = _next_shard_path(
        output_dir, prefix=args.output_prefix, shard_idx=shard_idx, gzip_output=bool(args.gzip)
    )
    writer = _open_text_writer(shard_path, gzip_output=bool(args.gzip))
    LOG.info("Writing shards to %s", output_dir)

    try:
        for i, path in enumerate(files):
            try:
                text = path.read_text(encoding="utf-8")
                ticket = json.loads(text)
            except Exception:
                parse_errors += 1
                if parse_errors <= 10:
                    LOG.warning("Failed to parse JSON: %s", path)
                progress.update(
                    processed_files=i + 1,
                    written_records=written,
                    parse_errors=parse_errors,
                    shard_idx=shard_idx,
                )
                continue

            if not isinstance(ticket, dict):
                parse_errors += 1
                if parse_errors <= 10:
                    LOG.warning("JSON is not an object: %s", path)
                progress.update(
                    processed_files=i + 1,
                    written_records=written,
                    parse_errors=parse_errors,
                    shard_idx=shard_idx,
                )
                continue

            if not args.keep_attachments:
                ticket.pop("attachments", None)

            ticket["source_path"] = _to_source_path(
                path, input_dir=input_dir, mode=args.source_path
            )

            writer.write(json.dumps(ticket, ensure_ascii=False))
            writer.write("\n")
            written += 1
            shard_records += 1

            progress.update(
                processed_files=i + 1,
                written_records=written,
                parse_errors=parse_errors,
                shard_idx=shard_idx,
            )

            if shard_records >= args.records_per_shard and (i + 1) < len(files):
                writer.close()
                shard_idx += 1
                shard_records = 0
                shard_path = _next_shard_path(
                    output_dir,
                    prefix=args.output_prefix,
                    shard_idx=shard_idx,
                    gzip_output=bool(args.gzip),
                )
                writer = _open_text_writer(shard_path, gzip_output=bool(args.gzip))

    finally:
        try:
            writer.close()
        except Exception:
            pass
        progress.update(
            processed_files=len(files),
            written_records=written,
            parse_errors=parse_errors,
            shard_idx=shard_idx,
        )
        progress.finish()

    LOG.info(
        "Done. Wrote %d ticket(s) into %d shard(s). Parse errors: %d.",
        written,
        shard_idx + 1,
        parse_errors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
