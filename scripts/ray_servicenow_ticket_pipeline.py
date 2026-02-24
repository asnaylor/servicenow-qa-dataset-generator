#!/usr/bin/env python3
"""
Ray Data pipeline for filtering and normalizing ServiceNow ticket JSON files.

This repo's JSON files (e.g. 81.json) are single JSON objects per file (pretty-printed),
not JSONL. Ray's `read_json` is often optimized for JSONL, so this pipeline reads files
as raw bytes and parses JSON in a map stage.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOG = logging.getLogger("servicenow_ticket_pipeline")


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _has_glob_magic(path: str) -> bool:
    return any(ch in path for ch in ("*", "?", "["))


def _discover_local_json_files(inputs: Sequence[str], recursive: bool) -> list[str]:
    files: list[str] = []
    for raw in inputs:
        if "://" in raw:
            raise ValueError(
                f"Remote URI inputs are not supported in this pipeline: {raw}"
            )

        p = Path(raw)
        if p.is_dir():
            pattern = "**/*.json" if recursive else "*.json"
            files.extend(str(x) for x in p.glob(pattern) if x.is_file())
            continue

        if _has_glob_magic(raw):
            files.extend(
                str(Path(x))
                for x in glob.glob(raw, recursive=recursive)
                if Path(x).is_file()
            )
            continue

        if p.is_file():
            files.append(str(p))
            continue

        raise FileNotFoundError(f"Input not found: {raw}")

    # Stable ordering helps reproducibility for local workflows.
    return sorted(files)


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return None


def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _norm_lower(value: Any) -> str:
    return _norm(value).lower()


@dataclass(frozen=True)
class RejectionConfig:
    require_closed_state: bool = True
    allowed_states: tuple[str, ...] = ("Closed",)
    require_inactive: bool = True
    allow_active_unknown: bool = False
    reject_auto_generated: bool = True

    # Heuristics for "auto generated". Keep these conservative and configurable.
    reject_sys_created_by: tuple[str, ...] = ("system",)
    reject_contact_type_substrings: tuple[str, ...] = (
        "event",
        "monitor",
        "alert",
        "integration",
        "auto",
    )
    allowed_alerting_rules: tuple[str, ...] = ("", "manual")
    reject_if_record_producer_present: bool = True
    reject_if_short_description_matches: tuple[str, ...] = (
        r"\bauto(?:-| )?generated\b",
        r"\bautomated\b",
    )


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str
    rejected_dir: str | None
    output_format: str  # "jsonl" (Ray write_json) or "parquet"
    overwrite: bool


def _ensure_output_dir(path: str, overwrite: bool) -> None:
    p = Path(path)
    if p.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _extract_discussion_texts(
    discussions: Mapping[str, Any] | None,
) -> dict[str, Any]:
    discussions = discussions or {}
    comments = discussions.get("customer_facing_comments") or []
    work_notes = discussions.get("internal_work_notes") or []

    def normalize_entries(entries: Any) -> list[dict[str, Any]]:
        if not isinstance(entries, list):
            return []
        normalized: list[dict[str, Any]] = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            normalized.append(
                {
                    "timestamp": _norm(e.get("timestamp")),
                    "created_by": _norm(e.get("created_by")),
                    "type": _norm(e.get("type")),
                    "text": _norm(e.get("text")),
                }
            )
        return normalized

    normalized_comments = normalize_entries(comments)
    normalized_work_notes = normalize_entries(work_notes)

    # Convenience fields for downstream NLP/QA extraction.
    joined_comments = "\n\n".join(
        t for t in (_norm(e.get("text")) for e in normalized_comments) if t
    )
    joined_work_notes = "\n\n".join(
        t for t in (_norm(e.get("text")) for e in normalized_work_notes) if t
    )

    return {
        "customer_comments": normalized_comments,
        "internal_work_notes": normalized_work_notes,
        "customer_comments_text": joined_comments,
        "internal_work_notes_text": joined_work_notes,
    }


def _evaluate_rejections(
    incident_fields: Mapping[str, Any],
    rejection_config: RejectionConfig,
) -> list[str]:
    reasons: list[str] = []

    state = _norm(incident_fields.get("state"))
    state_lower = state.lower()

    if rejection_config.require_closed_state:
        allowed = {s.lower() for s in rejection_config.allowed_states}
        if not state:
            reasons.append("missing_state")
        elif state_lower not in allowed:
            reasons.append(f"state_not_allowed:{state}")

    if rejection_config.require_inactive:
        active = _parse_bool(incident_fields.get("active"))
        if active is True:
            reasons.append("active_ticket")
        elif active is None:
            if not rejection_config.allow_active_unknown:
                reasons.append("active_unknown")

    if rejection_config.reject_auto_generated:
        sys_created_by = _norm_lower(incident_fields.get("sys_created_by"))
        if sys_created_by and sys_created_by in rejection_config.reject_sys_created_by:
            reasons.append("sys_created_by_system")

        contact_type = _norm_lower(incident_fields.get("contact_type"))
        if contact_type and any(
            token in contact_type for token in rejection_config.reject_contact_type_substrings
        ):
            reasons.append(f"contact_type_auto:{contact_type}")

        alerting_rule = _norm_lower(incident_fields.get("x_g_ner_alerting_rule"))
        if alerting_rule and alerting_rule not in rejection_config.allowed_alerting_rules:
            reasons.append(f"alerting_rule_auto:{alerting_rule}")

        if rejection_config.reject_if_record_producer_present:
            record_producer = _norm(incident_fields.get("u_record_producer"))
            if record_producer:
                reasons.append("record_producer_present")

        short_desc = _norm(incident_fields.get("short_description"))
        for pattern in rejection_config.reject_if_short_description_matches:
            if short_desc and re.search(pattern, short_desc, flags=re.IGNORECASE):
                reasons.append(f"short_description_matches:{pattern}")
                break

    # If we're requiring "closed", missing timestamps are a useful rejection reason.
    # Keep this distinct from state-based filtering (some workflows use Closed without closed_at).
    if rejection_config.require_closed_state and state_lower in {
        s.lower() for s in rejection_config.allowed_states
    }:
        closed_at = _norm(incident_fields.get("closed_at"))
        if not closed_at:
            reasons.append("missing_closed_at")

    return reasons


def _normalize_ticket_record(
    ticket: Mapping[str, Any],
    *,
    source_path: str,
    rejection_config: RejectionConfig,
) -> dict[str, Any]:
    metadata = ticket.get("metadata") or {}
    incident_fields = ticket.get("incident_fields") or {}
    discussions = ticket.get("discussions") or {}
    attachments = ticket.get("attachments") or []
    statistics = ticket.get("statistics") or {}

    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(incident_fields, dict):
        incident_fields = {}
    if not isinstance(discussions, dict):
        discussions = {}
    if not isinstance(statistics, dict):
        statistics = {}

    discussion_fields = _extract_discussion_texts(discussions)
    rejection_reasons = _evaluate_rejections(incident_fields, rejection_config)

    # Core identifiers.
    incident_number = _norm(metadata.get("incident_number")) or _norm(
        incident_fields.get("number")
    )
    sys_id = _norm(metadata.get("sys_id")) or _norm(incident_fields.get("sys_id"))

    # Keep only fields that are commonly useful for QA extraction.
    record: dict[str, Any] = {
        "kept": len(rejection_reasons) == 0,
        "rejection_reasons": rejection_reasons,
        "source_path": source_path,
        "table": _norm(metadata.get("table")),
        "export_timestamp": _norm(metadata.get("export_timestamp")),
        "incident_number": incident_number,
        "sys_id": sys_id,
        "number": _norm(incident_fields.get("number")),
        "state": _norm(incident_fields.get("state")),
        "active": _parse_bool(incident_fields.get("active")),
        "opened_at": _norm(incident_fields.get("opened_at") or incident_fields.get("sys_created_on")),
        "closed_at": _norm(incident_fields.get("closed_at")),
        "priority": _norm(incident_fields.get("priority")),
        "impact": _norm(incident_fields.get("impact")),
        "urgency": _norm(incident_fields.get("urgency")),
        "category": _norm(incident_fields.get("category")),
        "subcategory": _norm(incident_fields.get("subcategory")),
        "assignment_group": _norm(incident_fields.get("assignment_group")),
        "assigned_to": _norm(incident_fields.get("assigned_to")),
        "opened_by": _norm(incident_fields.get("opened_by")),
        "caller_id": _norm(incident_fields.get("caller_id")),
        "sys_created_by": _norm(incident_fields.get("sys_created_by")),
        "sys_updated_by": _norm(incident_fields.get("sys_updated_by")),
        "contact_type": _norm(incident_fields.get("contact_type")),
        "short_description": _norm(incident_fields.get("short_description")),
        "description": _norm(incident_fields.get("description")),
        "close_code": _norm(incident_fields.get("close_code")),
        "close_notes": _norm(incident_fields.get("close_notes")),
        "alerting_rule": _norm(incident_fields.get("x_g_ner_alerting_rule")),
        "record_producer": _norm(incident_fields.get("u_record_producer")),
        "total_comments": int(statistics.get("total_comments") or 0),
        "total_work_notes": int(statistics.get("total_work_notes") or 0),
        "total_attachments": int(statistics.get("total_attachments") or 0),
        "attachments_count": len(attachments) if isinstance(attachments, list) else 0,
        **discussion_fields,
    }

    return record


def _parse_and_normalize_row(
    row: Mapping[str, Any],
    *,
    rejection_config: RejectionConfig,
) -> dict[str, Any]:
    # Ray's `read_binary_files(..., include_paths=True)` has historically produced
    # columns named ("bytes", "path"), but different versions may vary. Handle both.
    source_path = _norm(row.get("path") or row.get("file_path") or row.get("uri"))
    blob = row.get("bytes") or row.get("data") or row.get("content")

    if blob is None:
        return {
            "kept": False,
            "rejection_reasons": ["missing_file_bytes"],
            "source_path": source_path,
        }

    if isinstance(blob, memoryview):
        blob = blob.tobytes()

    if isinstance(blob, str):
        text = blob
    else:
        try:
            text = blob.decode("utf-8")
        except Exception:
            return {
                "kept": False,
                "rejection_reasons": ["utf8_decode_error"],
                "source_path": source_path,
            }

    try:
        ticket = json.loads(text)
    except Exception:
        return {
            "kept": False,
            "rejection_reasons": ["json_parse_error"],
            "source_path": source_path,
        }

    if not isinstance(ticket, dict):
        return {
            "kept": False,
            "rejection_reasons": ["json_not_object"],
            "source_path": source_path,
        }

    return _normalize_ticket_record(
        ticket,
        source_path=source_path,
        rejection_config=rejection_config,
    )


def _iter_batch_rows(batch: Any) -> Iterable[Mapping[str, Any]]:
    """
    Yield row mappings from Ray Data `map_batches` inputs.

    Ray can provide batches in different formats depending on `batch_format`.
    This pipeline uses `batch_format="pyarrow"`, so the common input is a
    `pyarrow.Table`. Iterating over a pyarrow Table yields columns, not rows.
    """
    if isinstance(batch, list):
        for row in batch:
            if isinstance(row, Mapping):
                yield row
        return

    # pyarrow.Table
    if hasattr(batch, "to_pylist"):
        for row in batch.to_pylist():
            if isinstance(row, Mapping):
                yield row
        return

    # pandas.DataFrame
    if hasattr(batch, "to_dict") and hasattr(batch, "columns"):
        try:
            rows = batch.to_dict(orient="records")
        except TypeError:
            rows = batch.to_dict("records")
        for row in rows:
            if isinstance(row, Mapping):
                yield row
        return

    # dict-of-columns (numpy/pandas/pyarrow arrays)
    if isinstance(batch, Mapping):
        values = list(batch.values())
        if not values:
            return
        try:
            length = len(values[0])
        except Exception:
            return

        for idx in range(length):
            row: dict[str, Any] = {}
            for key, column in batch.items():
                try:
                    value = column[idx]
                except Exception:
                    continue
                if hasattr(value, "as_py"):
                    try:
                        value = value.as_py()
                    except Exception:
                        pass
                row[str(key)] = value
            yield row


def _columnarize_records(records: list[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """
    Convert per-row dict records into a dict-of-lists batch output.

    Ray 2.5.x `map_batches` does not accept returning `list[dict]` directly.
    Returning a dict-of-lists is a supported batch output shape.
    """
    if not records:
        return {"kept": [], "rejection_reasons": [], "source_path": []}

    all_keys: set[str] = set()
    for rec in records:
        for k in rec.keys():
            all_keys.add(str(k))

    # Stable order for reproducibility (and nicer debugging).
    keys = sorted(all_keys)
    out: dict[str, list[Any]] = {k: [] for k in keys}
    for rec in records:
        for k in keys:
            out[k].append(rec.get(k))
    return out


def _parse_and_normalize_batch(
    batch: Any,
    *,
    rejection_config: RejectionConfig,
) -> dict[str, list[Any]]:
    """
    Process a batch of rows for better performance with Ray Data.

    This uses map_batches which is more efficient than row-by-row map operations.
    """
    results: list[dict[str, Any]] = []
    for row in _iter_batch_rows(batch):
        results.append(_parse_and_normalize_row(row, rejection_config=rejection_config))
    return _columnarize_records(results)


def _write_dataset(ds: Any, *, output_dir: str, output_format: str) -> None:
    if output_format == "jsonl":
        ds.write_json(output_dir)
        return
    if output_format == "parquet":
        ds.write_parquet(output_dir)
        return
    raise ValueError(f"Unsupported output_format: {output_format}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and normalize ServiceNow ticket JSON files with Ray Data.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input file/dir/glob(s). For dirs, scans for *.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write kept/accepted tickets dataset.",
    )
    parser.add_argument(
        "--rejected-dir",
        default=None,
        help="Optional directory to write rejected tickets dataset.",
    )
    parser.add_argument(
        "--output-format",
        choices=("jsonl", "parquet"),
        default="jsonl",
        help="Output dataset format (Ray writer).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directories if they already exist.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse when --input points to directories or globs.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="For debugging: only process the first N discovered files.",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help='Ray address (e.g. "auto"). Omit to start local Ray.',
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent read tasks for Ray Data. Defaults to Ray auto-detection.",
    )

    # Rejection rules.
    parser.add_argument(
        "--allowed-state",
        action="append",
        default=[],
        help='Allowed state(s) for kept tickets. Repeatable. Default: "Closed".',
    )
    parser.add_argument(
        "--allow-active-unknown",
        action="store_true",
        help="Do not reject tickets when the 'active' field is missing/unparseable.",
    )
    parser.add_argument(
        "--keep-auto-generated",
        action="store_true",
        help="Do not apply auto-generated rejection heuristics.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    try:
        import ray  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Ray is required for this pipeline. Install with `pip install 'ray[data]'`.\n"
            f"Import error: {e}"
        )

    recursive = not args.no_recursive
    input_files = _discover_local_json_files(args.input, recursive=recursive)
    if args.limit_files is not None:
        input_files = input_files[: args.limit_files]
    if not input_files:
        raise SystemExit("No input files found.")
    LOG.info("Discovered %d input file(s).", len(input_files))

    output_config = OutputConfig(
        output_dir=args.output_dir,
        rejected_dir=args.rejected_dir,
        output_format=args.output_format,
        overwrite=bool(args.overwrite),
    )
    _ensure_output_dir(output_config.output_dir, output_config.overwrite)
    if output_config.rejected_dir:
        _ensure_output_dir(output_config.rejected_dir, output_config.overwrite)

    allowed_states = tuple(args.allowed_state) if args.allowed_state else ("Closed",)
    rejection_config = RejectionConfig(
        allowed_states=allowed_states,
        reject_auto_generated=not args.keep_auto_generated,
    )

    if args.allow_active_unknown:
        rejection_config = dataclasses.replace(rejection_config, allow_active_unknown=True)

    # Keep driver logs for debugging unless explicitly disabled
    ray.init(address=args.ray_address, ignore_reinit_error=True, log_to_driver=True)

    # Read and parse each JSON object per file with optional concurrency control
    read_kwargs = {"include_paths": True}
    if args.concurrency is not None:
        read_kwargs["concurrency"] = args.concurrency

    ds = ray.data.read_binary_files(input_files, **read_kwargs)

    # Use map_batches for better performance (2-5x faster than row-by-row map)
    LOG.info("Processing %d file(s) with Ray Data...", len(input_files))
    normalized = ds.map_batches(
        lambda batch: _parse_and_normalize_batch(batch, rejection_config=rejection_config),
        batch_format="pyarrow",
    )

    # Materialize (cache) the normalized dataset to avoid recomputing when filtering
    LOG.info("Normalizing and caching dataset...")
    normalized_cached = normalized.materialize()

    # Keep vs rejected datasets (single pass each after caching)
    kept = normalized_cached.filter(lambda r: bool(r.get("kept")))
    rejected = normalized_cached.filter(lambda r: not bool(r.get("kept")))

    # Write kept dataset
    LOG.info("Writing kept dataset to %s (%s).", output_config.output_dir, output_config.output_format)
    _write_dataset(
        kept.drop_columns(["kept"]),
        output_dir=output_config.output_dir,
        output_format=output_config.output_format,
    )

    # Write rejected dataset if requested
    if output_config.rejected_dir:
        LOG.info(
            "Writing rejected dataset to %s (%s).",
            output_config.rejected_dir,
            output_config.output_format,
        )
        _write_dataset(
            rejected.drop_columns(["kept"]),
            output_dir=output_config.rejected_dir,
            output_format=output_config.output_format,
        )

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
