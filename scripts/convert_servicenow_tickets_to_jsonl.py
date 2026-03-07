#!/usr/bin/env python3.11
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

import argparse
import dataclasses
import gzip
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Pattern
import tomllib


LOG = logging.getLogger("convert_servicenow_tickets_to_jsonl")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()

@dataclasses.dataclass(frozen=True)
class RejectionConfig:
    allowed_states: frozenset[str]
    allowed_close_codes: frozenset[str]
    rejected_close_codes: frozenset[str]
    allowed_contact_types: frozenset[str]
    rejected_contact_types: frozenset[str]
    short_description_patterns: tuple[Pattern[str], ...]
    allowed_categories: frozenset[str]
    rejected_categories: frozenset[str]
    allowed_resources: frozenset[str]
    rejected_resources: frozenset[str]
    rejected_assignment_groups: frozenset[str]
    min_opened_date: str  # ISO date string "YYYY-MM-DD", empty = no cutoff


def _load_toml_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _config_to_rejection_config(config: dict[str, Any]) -> RejectionConfig:
    def _lower_set(values: Any) -> frozenset[str]:
        if not isinstance(values, list):
            return frozenset()
        return frozenset(_norm(v).lower() for v in values if _norm(v))

    def _compiled_patterns(values: Any) -> tuple[Pattern[str], ...]:
        if not isinstance(values, list):
            return ()
        patterns: list[Pattern[str]] = []
        for value in values:
            pat = _norm(value)
            if not pat:
                continue
            patterns.append(re.compile(pat, flags=re.IGNORECASE))
        return tuple(patterns)

    state_cfg = config.get("state", {})
    close_codes_cfg = config.get("close_codes", {})
    contact_types_cfg = config.get("contact_types", {})
    auto_generated_cfg = config.get("auto_generated", {})
    categories_cfg = config.get("categories", {})
    resources_cfg = config.get("resources", {})
    assignment_groups_cfg = config.get("assignment_groups", {})
    date_cfg = config.get("date", {})

    return RejectionConfig(
        allowed_states=_lower_set(state_cfg.get("allowed_states", ["Closed", "Resolved"])),
        allowed_close_codes=_lower_set(close_codes_cfg.get("allowed", [])),
        rejected_close_codes=_lower_set(close_codes_cfg.get("rejected", [])),
        allowed_contact_types=_lower_set(contact_types_cfg.get("allowed", [])),
        rejected_contact_types=_lower_set(contact_types_cfg.get("rejected", [])),
        short_description_patterns=_compiled_patterns(
            auto_generated_cfg.get("reject_if_short_description_matches", [])
        ),
        allowed_categories=_lower_set(categories_cfg.get("allowed", [])),
        rejected_categories=_lower_set(categories_cfg.get("rejected", [])),
        allowed_resources=_lower_set(resources_cfg.get("allowed", [])),
        rejected_resources=_lower_set(resources_cfg.get("rejected", [])),
        rejected_assignment_groups=_lower_set(assignment_groups_cfg.get("rejected", [])),
        min_opened_date=_norm(date_cfg.get("min_opened_date", "")),
    )


def _should_reject(
    incident_fields: Mapping[str, Any],
    rejection_config: RejectionConfig,
) -> bool:
    # Date cutoff — reject tickets opened before min_opened_date.
    # ISO date strings compare lexicographically, so plain string comparison works.
    if rejection_config.min_opened_date:
        opened_at = _norm(incident_fields.get("opened_at"))
        if not opened_at or opened_at < rejection_config.min_opened_date:
            return True

    state = _norm(incident_fields.get("state"))
    state_lower = state.lower()
    if not state_lower or state_lower not in rejection_config.allowed_states:
        return True

    close_code = _norm(incident_fields.get("close_code"))
    close_code_lower = close_code.lower()
    if rejection_config.allowed_close_codes and (
        not close_code_lower or close_code_lower not in rejection_config.allowed_close_codes
    ):
        return True
    if close_code_lower and close_code_lower in rejection_config.rejected_close_codes:
        return True

    contact_type = _norm(incident_fields.get("contact_type"))
    contact_type_lower = contact_type.lower()
    if rejection_config.allowed_contact_types and (
        not contact_type_lower or contact_type_lower not in rejection_config.allowed_contact_types
    ):
        return True
    if contact_type_lower and contact_type_lower in rejection_config.rejected_contact_types:
        return True

    short_desc = _norm(incident_fields.get("short_description"))
    for pattern in rejection_config.short_description_patterns:
        if short_desc and pattern.search(short_desc):
            return True

    category = _norm(incident_fields.get("category"))
    category_lower = category.lower()
    if rejection_config.allowed_categories and (
        not category_lower or category_lower not in rejection_config.allowed_categories
    ):
        return True
    if category_lower and category_lower in rejection_config.rejected_categories:
        return True

    resource = _norm(incident_fields.get("u_resource"))
    resource_lower = resource.lower()
    if rejection_config.allowed_resources and (
        not resource_lower or resource_lower not in rejection_config.allowed_resources
    ):
        return True
    if resource_lower and resource_lower in rejection_config.rejected_resources:
        return True

    assignment_group = _norm(incident_fields.get("assignment_group"))
    if assignment_group.lower() in rejection_config.rejected_assignment_groups:
        return True

    return False


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
    default_config = Path(__file__).resolve().parents[1] / "config" / "qa_dataset.toml"
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
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to TOML config file for filtering rules (default: config/qa_dataset.toml).",
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
        filtered_records: int,
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
                f"filtered={filtered_records}  "
                f"errors={parse_errors}  "
                f"shards={shard_idx + 1}  "
                f"{rate:,.0f} files/s  "
                f"ETA {_format_eta(eta_s)}"
            )
            sys.stderr.write(msg)
            sys.stderr.flush()
        else:
            LOG.info(
                "Progress: %d/%d (%.1f%%) written=%d filtered=%d errors=%d shards=%d rate=%.0f files/s ETA=%s",
                processed_files,
                self._total_files,
                pct,
                written_records,
                filtered_records,
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
    LOG.info("Loading filtering config from: %s", args.config)
    filter_config = _load_toml_config(args.config)
    rejection_config = _config_to_rejection_config(filter_config)

    shard_idx = 0
    shard_records = 0
    written = 0
    filtered = 0
    parse_errors = 0

    progress = _Progress(
        total_files=len(files),
        enabled=not bool(args.no_progress),
        interval_s=float(args.progress_interval_seconds),
    )

    writer = None
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
                    filtered_records=filtered,
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
                    filtered_records=filtered,
                    parse_errors=parse_errors,
                    shard_idx=shard_idx,
                )
                continue

            incident_fields = ticket.get("incident_fields")
            if not isinstance(incident_fields, dict):
                incident_fields = {}
            if _should_reject(incident_fields, rejection_config):
                filtered += 1
                progress.update(
                    processed_files=i + 1,
                    written_records=written,
                    filtered_records=filtered,
                    parse_errors=parse_errors,
                    shard_idx=shard_idx,
                )
                continue

            if not args.keep_attachments:
                ticket.pop("attachments", None)

            ticket["source_path"] = _to_source_path(
                path, input_dir=input_dir, mode=args.source_path
            )

            if writer is None:
                shard_path = _next_shard_path(
                    output_dir,
                    prefix=args.output_prefix,
                    shard_idx=shard_idx,
                    gzip_output=bool(args.gzip),
                )
                writer = _open_text_writer(shard_path, gzip_output=bool(args.gzip))

            writer.write(json.dumps(ticket, ensure_ascii=False))
            writer.write("\n")
            written += 1
            shard_records += 1

            progress.update(
                processed_files=i + 1,
                written_records=written,
                filtered_records=filtered,
                parse_errors=parse_errors,
                shard_idx=shard_idx,
            )

            if shard_records >= args.records_per_shard and (i + 1) < len(files):
                writer.close()
                writer = None
                shard_idx += 1
                shard_records = 0

    finally:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        progress.update(
            processed_files=len(files),
            written_records=written,
            filtered_records=filtered,
            parse_errors=parse_errors,
            shard_idx=shard_idx,
        )
        progress.finish()

    LOG.info(
        "Done. Wrote %d ticket(s) into %d shard(s). Filtered out: %d. Parse errors: %d.",
        written,
        (shard_idx + 1) if written else 0,
        filtered,
        parse_errors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
