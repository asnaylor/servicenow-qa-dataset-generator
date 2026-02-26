#!/usr/bin/env python3
"""
Ray Data pipeline for filtering and normalizing ServiceNow ticket JSON files.

This pipeline expects JSONL shards (one ticket per line). Use
`scripts/convert_servicenow_tickets_to_jsonl.py` to convert from "one JSON object per
file" exports into JSONL.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


LOG = logging.getLogger("servicenow_ticket_pipeline")


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _has_glob_magic(path: str) -> bool:
    return any(ch in path for ch in ("*", "?", "["))


def _discover_local_files(
    inputs: Sequence[str],
    *,
    recursive: bool,
    allowed_suffixes: tuple[str, ...],
) -> list[str]:
    files: list[str] = []

    def is_allowed(path: Path) -> bool:
        s = str(path)
        return any(s.endswith(suffix) for suffix in allowed_suffixes)

    for raw in inputs:
        if "://" in raw:
            raise ValueError(
                f"Remote URI inputs are not supported in this pipeline: {raw}"
            )

        p = Path(raw)
        if p.is_dir():
            for suffix in allowed_suffixes:
                pattern = f"**/*{suffix}" if recursive else f"*{suffix}"
                files.extend(str(x) for x in p.glob(pattern) if x.is_file())
            continue

        if _has_glob_magic(raw):
            files.extend(
                str(Path(x))
                for x in glob.glob(raw, recursive=recursive)
                if Path(x).is_file() and is_allowed(Path(x))
            )
            continue

        if p.is_file():
            if not is_allowed(p):
                raise ValueError(
                    f"Input file does not match expected suffixes {allowed_suffixes}: {raw}"
                )
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
    allowed_states: tuple[str, ...] = ("Closed", "Resolved")
    require_inactive: bool = True
    allow_active_unknown: bool = False
    reject_auto_generated: bool = True
    require_close_code: bool = False
    allowed_close_codes: tuple[str, ...] = ()
    reject_close_codes: tuple[str, ...] = ()

    # Heuristics for "auto generated". Keep these conservative and configurable.
    reject_sys_created_by: tuple[str, ...] = (
        "system",
        "autoticketing",
        "auto-ops",
        "auto-consult",
        "auto-consultant",
        "home_perm_tickets",
    )
    reject_contact_type_substrings: tuple[str, ...] = (
        "event",
        "monitor",
        "alert",
        "integration",
        "auto",
    )
    allowed_alerting_rules: tuple[str, ...] = ("", "manual", "default")
    reject_if_record_producer_present: bool = True
    reject_if_short_description_matches: tuple[str, ...] = (
        r"\bauto(?:-| )?generated\b",
        r"\bautomated\b",
    )

    # Content quality filters for Q&A datasets
    require_close_notes: bool = False
    min_short_description_length: int = 0
    min_close_notes_length: int = 0
    min_total_comments: int = 0
    reject_generic_close_notes: bool = False
    reject_generic_close_notes_patterns: tuple[str, ...] = ()

    # Contact type filtering
    allowed_contact_types: tuple[str, ...] = ()
    reject_contact_types: tuple[str, ...] = ()

    # Category/resource filtering
    allowed_categories: tuple[str, ...] = ()
    reject_categories: tuple[str, ...] = ()
    allowed_resources: tuple[str, ...] = ()
    reject_resources: tuple[str, ...] = ()


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
        # If `path` is a bind mount (common in containers), attempting to remove the
        # mountpoint itself can fail with "Device or resource busy". Instead, clear
        # contents in-place and keep the directory.
        if p.is_dir():
            for child in p.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            p.unlink()
            p.mkdir(parents=True, exist_ok=True)
            return
    else:
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
    allowed_states_lower = {s.lower() for s in rejection_config.allowed_states}
    is_allowed_state = bool(state_lower) and state_lower in allowed_states_lower
    close_code = _norm(incident_fields.get("close_code"))
    close_code_lower = close_code.lower()

    if rejection_config.require_closed_state:
        if not state:
            reasons.append("missing_state")
        elif not is_allowed_state:
            reasons.append(f"state_not_allowed:{state}")

    if rejection_config.require_close_code and not close_code:
        reasons.append("missing_close_code")

    if rejection_config.allowed_close_codes:
        allowed_close_codes_lower = {c.lower() for c in rejection_config.allowed_close_codes}
        if close_code_lower and close_code_lower not in allowed_close_codes_lower:
            reasons.append(f"close_code_not_allowed:{close_code}")

    if rejection_config.reject_close_codes:
        reject_close_codes_lower = {c.lower() for c in rejection_config.reject_close_codes}
        if close_code_lower and close_code_lower in reject_close_codes_lower:
            reasons.append(f"close_code_rejected:{close_code}")

    if rejection_config.require_inactive:
        active = _parse_bool(incident_fields.get("active"))
        if active is True:
            # In ServiceNow exports, "Resolved" tickets are often still marked active=true.
            # If the state is allowed and terminal, prefer the state field over `active`.
            if not (is_allowed_state and state_lower == "resolved"):
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
    if rejection_config.require_closed_state and is_allowed_state:
        closed_at = _norm(incident_fields.get("closed_at"))
        resolved_at = _norm(incident_fields.get("resolved_at"))
        if state_lower == "closed":
            if not closed_at:
                reasons.append("missing_closed_at")
        else:
            if not closed_at and not resolved_at:
                reasons.append("missing_closed_or_resolved_at")

    # Content quality filters
    if rejection_config.require_close_notes:
        close_notes = _norm(incident_fields.get("close_notes"))
        if not close_notes:
            reasons.append("missing_close_notes")

    if rejection_config.min_short_description_length > 0:
        short_desc = _norm(incident_fields.get("short_description"))
        if len(short_desc) < rejection_config.min_short_description_length:
            reasons.append(f"short_description_too_short:{len(short_desc)}")

    if rejection_config.min_close_notes_length > 0:
        close_notes = _norm(incident_fields.get("close_notes"))
        if len(close_notes) < rejection_config.min_close_notes_length:
            reasons.append(f"close_notes_too_short:{len(close_notes)}")

    if rejection_config.reject_generic_close_notes:
        close_notes = _norm(incident_fields.get("close_notes"))
        for pattern in rejection_config.reject_generic_close_notes_patterns:
            if close_notes and re.search(pattern, close_notes, flags=re.IGNORECASE):
                reasons.append("close_notes_generic")
                break

    # Contact type filtering
    contact_type = _norm(incident_fields.get("contact_type"))
    contact_type_lower = contact_type.lower()
    if rejection_config.allowed_contact_types:
        allowed_contact_types_lower = {c.lower() for c in rejection_config.allowed_contact_types}
        if contact_type_lower and contact_type_lower not in allowed_contact_types_lower:
            reasons.append(f"contact_type_not_allowed:{contact_type}")

    if rejection_config.reject_contact_types:
        reject_contact_types_lower = {c.lower() for c in rejection_config.reject_contact_types}
        if contact_type_lower and contact_type_lower in reject_contact_types_lower:
            reasons.append(f"contact_type_rejected:{contact_type}")

    # Category filtering
    category = _norm(incident_fields.get("category"))
    category_lower = category.lower()
    if rejection_config.allowed_categories:
        allowed_categories_lower = {c.lower() for c in rejection_config.allowed_categories}
        if category_lower and category_lower not in allowed_categories_lower:
            reasons.append(f"category_not_allowed:{category}")

    if rejection_config.reject_categories:
        reject_categories_lower = {c.lower() for c in rejection_config.reject_categories}
        if category_lower and category_lower in reject_categories_lower:
            reasons.append(f"category_rejected:{category}")

    # Resource filtering
    resource = _norm(incident_fields.get("u_resource"))
    resource_lower = resource.lower()
    if rejection_config.allowed_resources:
        allowed_resources_lower = {r.lower() for r in rejection_config.allowed_resources}
        if resource_lower and resource_lower not in allowed_resources_lower:
            reasons.append(f"resource_not_allowed:{resource}")

    if rejection_config.reject_resources:
        reject_resources_lower = {r.lower() for r in rejection_config.reject_resources}
        if resource_lower and resource_lower in reject_resources_lower:
            reasons.append(f"resource_rejected:{resource}")

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

    # Check min_total_comments (statistics-based filter)
    if rejection_config.min_total_comments > 0:
        total_comments = int(statistics.get("total_comments") or 0)
        if total_comments < rejection_config.min_total_comments:
            rejection_reasons.append(f"total_comments_too_few:{total_comments}")

    total_attachments = int(statistics.get("total_attachments") or 0)
    attachments_count = len(attachments) if isinstance(attachments, list) else 0
    # If attachments were dropped upstream (e.g. during JSONL conversion), preserve
    # a reasonable count from the statistics block.
    if attachments_count == 0 and total_attachments:
        attachments_count = total_attachments

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
        "resolved_at": _norm(incident_fields.get("resolved_at")),
        "priority": _norm(incident_fields.get("priority")),
        "impact": _norm(incident_fields.get("impact")),
        "urgency": _norm(incident_fields.get("urgency")),
        "category": _norm(incident_fields.get("category")),
        "subcategory": _norm(incident_fields.get("subcategory")),
        "u_resource": _norm(incident_fields.get("u_resource")),
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
        "total_attachments": total_attachments,
        "attachments_count": attachments_count,
        **discussion_fields,
    }

    return record


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


def _normalize_ticket_batch(
    batch: Any,
    *,
    rejection_config: RejectionConfig,
) -> dict[str, list[Any]]:
    """
    Normalize a batch of already-parsed ticket dicts (e.g. from `read_json` on JSONL).
    """
    results: list[dict[str, Any]] = []
    for row in _iter_batch_rows(batch):
        source_path = _norm(row.get("source_path") or row.get("path") or "")
        # `row` is the ticket dict when reading JSONL (plus optional provenance fields).
        results.append(
            _normalize_ticket_record(
                row,
                source_path=source_path,
                rejection_config=rejection_config,
            )
        )
    return _columnarize_records(results)


def _write_dataset(ds: Any, *, output_dir: str, output_format: str) -> None:
    if output_format == "jsonl":
        ds.write_json(output_dir)
        return
    if output_format == "parquet":
        ds.write_parquet(output_dir)
        return
    raise ValueError(f"Unsupported output_format: {output_format}")


def _load_toml_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a TOML file."""
    if tomllib is None:
        raise SystemExit(
            "TOML support not available. Install with `pip install tomli` (Python <3.11) "
            "or use Python 3.11+ which includes tomllib."
        )

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _config_to_rejection_config(config: dict[str, Any]) -> RejectionConfig:
    """Convert TOML config dict to RejectionConfig."""
    state_cfg = config.get("state", {})
    close_codes_cfg = config.get("close_codes", {})
    contact_types_cfg = config.get("contact_types", {})
    auto_generated_cfg = config.get("auto_generated", {})
    content_quality_cfg = config.get("content_quality", {})
    categories_cfg = config.get("categories", {})
    resources_cfg = config.get("resources", {})

    return RejectionConfig(
        require_closed_state=state_cfg.get("require_closed_state", True),
        allowed_states=tuple(state_cfg.get("allowed_states", ["Closed", "Resolved"])),
        require_inactive=state_cfg.get("require_inactive", True),
        allow_active_unknown=state_cfg.get("allow_active_unknown", False),
        reject_auto_generated=auto_generated_cfg.get("enabled", True),
        require_close_code=bool(close_codes_cfg.get("allowed") or close_codes_cfg.get("rejected")),
        allowed_close_codes=tuple(close_codes_cfg.get("allowed", [])),
        reject_close_codes=tuple(close_codes_cfg.get("rejected", [])),
        reject_sys_created_by=tuple(auto_generated_cfg.get("reject_sys_created_by", [])),
        reject_contact_type_substrings=tuple(auto_generated_cfg.get("reject_contact_type_substrings", [])),
        allowed_alerting_rules=tuple(auto_generated_cfg.get("allowed_alerting_rules", ["", "manual", "default"])),
        reject_if_record_producer_present=auto_generated_cfg.get("reject_if_record_producer_present", True),
        reject_if_short_description_matches=tuple(auto_generated_cfg.get("reject_if_short_description_matches", [])),
        require_close_notes=content_quality_cfg.get("require_close_notes", False),
        min_short_description_length=content_quality_cfg.get("min_short_description_length", 0),
        min_close_notes_length=content_quality_cfg.get("min_close_notes_length", 0),
        min_total_comments=content_quality_cfg.get("min_total_comments", 0),
        reject_generic_close_notes=content_quality_cfg.get("reject_generic_close_notes", False),
        reject_generic_close_notes_patterns=tuple(
            content_quality_cfg.get("reject_generic_close_notes_patterns", [])
        ),
        allowed_contact_types=tuple(contact_types_cfg.get("allowed", [])),
        reject_contact_types=tuple(contact_types_cfg.get("rejected", [])),
        allowed_categories=tuple(categories_cfg.get("allowed", [])),
        reject_categories=tuple(categories_cfg.get("rejected", [])),
        allowed_resources=tuple(resources_cfg.get("allowed", [])),
        reject_resources=tuple(resources_cfg.get("rejected", [])),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and normalize ServiceNow ticket JSON files with Ray Data.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to TOML config file for filtering rules.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help='Input JSONL file/dir/glob(s). For dirs, scans for "*.jsonl" and "*.jsonl.gz".',
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
        "--read-num-blocks",
        type=int,
        default=None,
        help=(
            "Controls JSONL read parallelism by setting `override_num_blocks` on "
            "ray.data.read_json (Ray 2.53.0). If omitted, the script picks an "
            "automatic value based on cluster CPUs and total input size."
        ),
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

    input_files = _discover_local_files(
        args.input,
        recursive=recursive,
        allowed_suffixes=(".jsonl", ".jsonl.gz"),
    )

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

    # Load config from TOML
    LOG.info("Loading configuration from: %s", args.config)
    toml_config = _load_toml_config(args.config)
    rejection_config = _config_to_rejection_config(toml_config)

    # Keep driver logs for debugging unless explicitly disabled
    ray.init(address=args.ray_address, ignore_reinit_error=True, log_to_driver=True)

    # JSONL shards: one JSON object per line.
    # Automatic default: aim to create enough tasks to utilize the cluster while
    # avoiding extremely tiny blocks.
    total_bytes = 0
    for raw in input_files:
        try:
            total_bytes += Path(raw).stat().st_size
        except Exception:
            # Best-effort only; fall back to CPU-based heuristics.
            total_bytes = 0
            break

    cluster_cpus = int(ray.cluster_resources().get("CPU", 1))
    requested_num_blocks = args.read_num_blocks

    min_block_bytes = 4 * 1024 * 1024  # 4 MiB: avoid creating extremely tiny blocks
    max_blocks_by_size = (
        max(1, (total_bytes + min_block_bytes - 1) // min_block_bytes)
        if total_bytes
        else None
    )
    auto_num_blocks = max(cluster_cpus, len(input_files) * 4)
    if max_blocks_by_size is not None:
        auto_num_blocks = min(auto_num_blocks, int(max_blocks_by_size))

    chosen_num_blocks = (
        int(requested_num_blocks) if requested_num_blocks is not None else int(auto_num_blocks)
    )

    # Ray 2.53.0: `ray.data.read_json` supports `override_num_blocks`.
    read_kwargs: dict[str, Any] = {"override_num_blocks": chosen_num_blocks}
    if requested_num_blocks is None:
        LOG.info(
            "Auto-selecting JSONL read parallelism: override_num_blocks=%d (cluster CPUs=%d, shards=%d, size≈%.2f GiB).",
            chosen_num_blocks,
            cluster_cpus,
            len(input_files),
            (total_bytes / (1024**3)) if total_bytes else 0.0,
        )
    else:
        LOG.info(
            "Using requested JSONL read parallelism: override_num_blocks=%d.",
            chosen_num_blocks,
        )

    ds = ray.data.read_json(input_files, **read_kwargs)

    LOG.info("Processing %d JSONL shard(s) with Ray Data...", len(input_files))
    normalized = ds.map_batches(
        lambda batch: _normalize_ticket_batch(batch, rejection_config=rejection_config),
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
