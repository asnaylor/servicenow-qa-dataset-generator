#!/usr/bin/env python3
"""
MVP Ray Data LLM pipeline for ServiceNow QA dataset generation.

Flow:
1) Read filtered JSONL ticket shards (output of convert_servicenow_tickets_to_jsonl.py).
2) LLM quality gate: reject tickets that are not useful for QA synthesis.
3) LLM synthesis: generate structured Q&A JSON.
4) Write accepted Q&A records to Parquet.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[import-not-found,no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

LOG = logging.getLogger("ray_servicenow_ticket_pipeline")


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
            raise ValueError(f"Remote URI inputs are not supported: {raw}")

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

    return sorted(files)


def _ensure_output_dir(path: str, overwrite: bool) -> None:
    p = Path(path)
    if p.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
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


def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _join_text_entries(entries: Any) -> str:
    if not isinstance(entries, list):
        return ""
    out: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = _norm(entry.get("text"))
        if text:
            out.append(text)
    return "\n\n".join(out)


def _ticket_to_context(ticket: Mapping[str, Any], *, max_chars: int) -> str:
    incident = ticket.get("incident_fields") or {}
    discussions = ticket.get("discussions") or {}
    metadata = ticket.get("metadata") or {}
    statistics = ticket.get("statistics") or {}

    if not isinstance(incident, dict):
        incident = {}
    if not isinstance(discussions, dict):
        discussions = {}
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(statistics, dict):
        statistics = {}

    comments_text = _join_text_entries(discussions.get("customer_facing_comments"))
    notes_text = _join_text_entries(discussions.get("internal_work_notes"))

    context = (
        f"Incident Number: {_norm(incident.get('number') or metadata.get('incident_number'))}\n"
        f"State: {_norm(incident.get('state'))}\n"
        f"Category: {_norm(incident.get('category'))}\n"
        f"Subcategory: {_norm(incident.get('subcategory'))}\n"
        f"Resource: {_norm(incident.get('u_resource'))}\n"
        f"Short Description: {_norm(incident.get('short_description'))}\n"
        f"Description:\n{_norm(incident.get('description'))}\n\n"
        f"Close Code: {_norm(incident.get('close_code'))}\n"
        f"Close Notes:\n{_norm(incident.get('close_notes'))}\n\n"
        f"Customer Comments:\n{comments_text}\n\n"
        f"Internal Work Notes:\n{notes_text}\n\n"
        f"Stats: total_comments={int(statistics.get('total_comments') or 0)}, "
        f"total_work_notes={int(statistics.get('total_work_notes') or 0)}"
    )
    if len(context) > max_chars:
        return context[:max_chars]
    return context


def _to_llm_input_record(ticket: dict[str, Any], *, context_max_chars: int) -> dict[str, Any]:
    incident = ticket.get("incident_fields") or {}
    metadata = ticket.get("metadata") or {}
    if not isinstance(incident, dict):
        incident = {}
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "source_path": _norm(ticket.get("source_path")),
        "incident_number": _norm(incident.get("number") or metadata.get("incident_number")),
        "sys_id": _norm(incident.get("sys_id") or metadata.get("sys_id")),
        "state": _norm(incident.get("state")),
        "category": _norm(incident.get("category")),
        "subcategory": _norm(incident.get("subcategory")),
        "u_resource": _norm(incident.get("u_resource")),
        "short_description": _norm(incident.get("short_description")),
        "close_code": _norm(incident.get("close_code")),
        "ticket_context": _ticket_to_context(ticket, max_chars=context_max_chars),
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    s = text.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _build_quality_preprocess(
    *,
    temperature: float,
    max_tokens: int,
    context_chars: int,
):
    system_prompt = (
        "You are a strict QA dataset curator for IT support tickets. "
        "Decide if a resolved ticket is useful for training question-answering. "
        "Useful tickets have enough concrete problem and resolution detail."
    )

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        context = _norm(row.get("ticket_context"))[:context_chars]
        user_prompt = (
            "Return ONLY valid JSON with keys keep (boolean) and reason (string).\n"
            "Reject if details are too sparse, ambiguous, or not actionable.\n\n"
            f"Ticket:\n{context}"
        )
        return {
            "source_path": _norm(row.get("source_path")),
            "incident_number": _norm(row.get("incident_number")),
            "sys_id": _norm(row.get("sys_id")),
            "state": _norm(row.get("state")),
            "category": _norm(row.get("category")),
            "subcategory": _norm(row.get("subcategory")),
            "u_resource": _norm(row.get("u_resource")),
            "short_description": _norm(row.get("short_description")),
            "close_code": _norm(row.get("close_code")),
            "ticket_context": context,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "sampling_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }

    return preprocess


def _quality_postprocess(row: dict[str, Any]) -> dict[str, Any]:
    raw = _norm(row.get("generated_text"))
    parsed = _extract_json_object(raw)
    keep = False
    reason = "invalid_json_from_llm"
    if parsed is not None:
        keep = bool(parsed.get("keep", False))
        reason = _norm(parsed.get("reason")) or reason

    return {
        **row,
        "quality_keep": keep,
        "quality_reason": reason,
        "quality_raw": raw,
    }


def _build_qa_preprocess(
    *,
    temperature: float,
    max_tokens: int,
    context_chars: int,
):
    system_prompt = (
        "You are an expert technical writer. Convert a ServiceNow incident ticket "
        "into a concise synthetic Q&A pair for training."
    )

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        context = _norm(row.get("ticket_context"))[:context_chars]
        user_prompt = (
            "Return ONLY valid JSON with keys:\n"
            "- question (string)\n"
            "- answer (string)\n"
            "- summary (string)\n"
            "- tags (array of strings)\n"
            "The answer must be grounded in the ticket details.\n\n"
            f"Ticket:\n{context}"
        )
        return {
            "source_path": _norm(row.get("source_path")),
            "incident_number": _norm(row.get("incident_number")),
            "sys_id": _norm(row.get("sys_id")),
            "state": _norm(row.get("state")),
            "category": _norm(row.get("category")),
            "subcategory": _norm(row.get("subcategory")),
            "u_resource": _norm(row.get("u_resource")),
            "short_description": _norm(row.get("short_description")),
            "close_code": _norm(row.get("close_code")),
            "quality_reason": _norm(row.get("quality_reason")),
            "ticket_context": context,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "sampling_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }

    return preprocess


def _qa_postprocess(row: dict[str, Any]) -> dict[str, Any]:
    raw = _norm(row.get("generated_text"))
    parsed = _extract_json_object(raw)

    if parsed is None:
        return {
            **row,
            "qa_parse_ok": False,
            "qa_question": "",
            "qa_answer": "",
            "qa_summary": "",
            "qa_tags": [],
            "qa_raw": raw,
        }

    tags = parsed.get("tags")
    if not isinstance(tags, list):
        tags = []
    tag_values = [_norm(x) for x in tags if _norm(x)]

    return {
        **row,
        "qa_parse_ok": True,
        "qa_question": _norm(parsed.get("question")),
        "qa_answer": _norm(parsed.get("answer")),
        "qa_summary": _norm(parsed.get("summary")),
        "qa_tags": tag_values,
        "qa_raw": raw,
    }


def _to_output_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_path": _norm(row.get("source_path")),
        "incident_number": _norm(row.get("incident_number")),
        "sys_id": _norm(row.get("sys_id")),
        "state": _norm(row.get("state")),
        "category": _norm(row.get("category")),
        "subcategory": _norm(row.get("subcategory")),
        "u_resource": _norm(row.get("u_resource")),
        "short_description": _norm(row.get("short_description")),
        "close_code": _norm(row.get("close_code")),
        "qa_question": _norm(row.get("qa_question")),
        "qa_answer": _norm(row.get("qa_answer")),
        "qa_summary": _norm(row.get("qa_summary")),
        "qa_tags": row.get("qa_tags") or [],
        "quality_reason": _norm(row.get("quality_reason")),
    }


def _load_toml_config(config_path: str) -> dict[str, Any]:
    if tomllib is None:
        raise SystemExit(
            "TOML support not available. Install `tomli` for Python <3.11, or use Python 3.11+."
        )
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _count_alive_nodes(ray_module: Any) -> int:
    try:
        return max(1, sum(1 for n in ray_module.nodes() if n.get("Alive")))
    except Exception:
        return 1


def _resolve_auto_concurrency(ray_module: Any, *, tensor_parallel_size: int) -> int:
    tp = max(1, int(tensor_parallel_size))
    try:
        resources = ray_module.available_resources()
        available_gpus = float(resources.get("GPU", 0.0))
        gpu_replicas = int(available_gpus // float(tp))
        if gpu_replicas > 0:
            return gpu_replicas
    except Exception:
        pass
    return _count_alive_nodes(ray_module)


def _resolve_concurrency(
    llm_cfg: Mapping[str, Any],
    ray_module: Any,
    *,
    tensor_parallel_size: int,
) -> int:
    raw = llm_cfg.get("concurrency", "auto")
    if isinstance(raw, str):
        if raw.strip().lower() == "auto":
            return _resolve_auto_concurrency(
                ray_module,
                tensor_parallel_size=tensor_parallel_size,
            )
        return max(1, int(raw))
    if isinstance(raw, (int, float)):
        if int(raw) <= 0:
            return _resolve_auto_concurrency(
                ray_module,
                tensor_parallel_size=tensor_parallel_size,
            )
        return int(raw)
    return _resolve_auto_concurrency(
        ray_module,
        tensor_parallel_size=tensor_parallel_size,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_llm_config = Path(__file__).resolve().parents[1] / "config" / "llm_pipeline.toml"
    parser = argparse.ArgumentParser(
        description="Run LLM quality filter + QA synthesis over filtered ServiceNow JSONL.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help='Input JSONL file/dir/glob(s). For dirs, scans for "*.jsonl" and "*.jsonl.gz".',
    )
    parser.add_argument("--output-dir", required=True, help="Output Parquet directory for accepted QA.")
    parser.add_argument(
        "--rejected-dir",
        default=None,
        help="Optional output Parquet directory for LLM-quality rejected tickets.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directories.")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive file discovery.")
    parser.add_argument(
        "--limit-tickets",
        type=int,
        default=None,
        help="Only process the first N ticket records after reading JSONL.",
    )
    parser.add_argument("--ray-address", default=None, help='Ray address (e.g. "auto").')
    parser.add_argument(
        "--llm-config",
        default=str(default_llm_config),
        help="Path to LLM TOML config (default: config/llm_pipeline.toml).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    try:
        import ray  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Ray is required. Install with `pip install ray[data]`.\n"
            f"Import error: {e}"
        )

    try:
        from ray.data.llm import vLLMEngineProcessorConfig
        try:
            from ray.data.llm import build_processor
        except Exception:
            from ray.data.llm import build_llm_processor as build_processor
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Ray Data LLM API is required. Ensure your Ray version supports `ray.data.llm`.\n"
            f"Import error: {e}"
        )

    recursive = not args.no_recursive
    input_files = _discover_local_files(
        args.input,
        recursive=recursive,
        allowed_suffixes=(".jsonl", ".jsonl.gz"),
    )
    if not input_files:
        raise SystemExit("No input files found.")

    _ensure_output_dir(args.output_dir, bool(args.overwrite))
    if args.rejected_dir:
        _ensure_output_dir(args.rejected_dir, bool(args.overwrite))

    llm_cfg = _load_toml_config(args.llm_config)
    LOG.info("Loaded LLM config from: %s", args.llm_config)
    model_source = str(llm_cfg.get("model_source", "meta-llama/Llama-3.1-8B-Instruct"))
    batch_size = int(llm_cfg.get("batch_size", 32))
    max_model_len = int(llm_cfg.get("max_model_len", 8192))
    context_max_chars = int(llm_cfg.get("context_max_chars", 12000))

    quality_cfg = llm_cfg.get("quality", {})
    if not isinstance(quality_cfg, dict):
        quality_cfg = {}
    quality_temperature = float(quality_cfg.get("temperature", 0.0))
    quality_max_tokens = int(quality_cfg.get("max_tokens", 128))

    qa_cfg = llm_cfg.get("qa", {})
    if not isinstance(qa_cfg, dict):
        qa_cfg = {}
    qa_temperature = float(qa_cfg.get("temperature", 0.1))
    qa_max_tokens = int(qa_cfg.get("max_tokens", 512))

    engine_kwargs_cfg = llm_cfg.get("engine_kwargs", {})
    if not isinstance(engine_kwargs_cfg, dict):
        engine_kwargs_cfg = {}
    base_engine_kwargs: dict[str, Any] = dict(engine_kwargs_cfg)
    if "max_model_len" not in base_engine_kwargs:
        base_engine_kwargs["max_model_len"] = max_model_len

    tensor_parallel_size = max(1, int(base_engine_kwargs.get("tensor_parallel_size", 1)))

    LOG.info("Discovered %d JSONL input file(s).", len(input_files))
    ray.init(address=args.ray_address, ignore_reinit_error=True, log_to_driver=True)
    concurrency = _resolve_concurrency(
        llm_cfg,
        ray,
        tensor_parallel_size=tensor_parallel_size,
    )

    ds = ray.data.read_json(input_files)
    if args.limit_tickets is not None:
        ds = ds.limit(int(args.limit_tickets))
    ds = ds.map(lambda row: _to_llm_input_record(row, context_max_chars=context_max_chars))
    ds = ds.materialize()
    total_count = ds.count()
    LOG.info("Loaded %d ticket record(s).", total_count)

    LOG.info("Using vLLM engine kwargs: %s", base_engine_kwargs)
    LOG.info("Using LLM concurrency=%d", concurrency)
    quality_engine_config_kwargs: dict[str, Any] = {
        "model_source": model_source,
        "engine_kwargs": base_engine_kwargs,
        "concurrency": concurrency,
        "batch_size": batch_size,
    }
    quality_processor = build_processor(
        config=vLLMEngineProcessorConfig(**quality_engine_config_kwargs),
        preprocess=_build_quality_preprocess(
            temperature=quality_temperature,
            max_tokens=quality_max_tokens,
            context_chars=context_max_chars,
        ),
        postprocess=_quality_postprocess,
    )

    LOG.info("Running LLM quality filter...")
    quality_ds = quality_processor(ds).materialize()

    kept_ds = quality_ds.filter(lambda row: bool(row.get("quality_keep")))
    rejected_ds = quality_ds.filter(lambda row: not bool(row.get("quality_keep")))
    kept_count = kept_ds.count()
    rejected_count = rejected_ds.count()
    LOG.info(
        "Quality filter complete. kept=%d rejected=%d",
        kept_count,
        rejected_count,
    )

    if args.rejected_dir:
        rejected_out = rejected_ds.map(
            lambda row: {
                "source_path": _norm(row.get("source_path")),
                "quality_reason": _norm(row.get("quality_reason")),
                "quality_raw": _norm(row.get("quality_raw")),
            }
        )
        rejected_out.write_parquet(args.rejected_dir)
        LOG.info("Wrote rejected quality records to %s", args.rejected_dir)

    qa_engine_config_kwargs: dict[str, Any] = {
        "model_source": model_source,
        "engine_kwargs": base_engine_kwargs,
        "concurrency": concurrency,
        "batch_size": batch_size,
    }
    qa_processor = build_processor(
        config=vLLMEngineProcessorConfig(**qa_engine_config_kwargs),
        preprocess=_build_qa_preprocess(
            temperature=qa_temperature,
            max_tokens=qa_max_tokens,
            context_chars=context_max_chars,
        ),
        postprocess=_qa_postprocess,
    )

    LOG.info("Running LLM Q&A synthesis for kept tickets...")
    qa_ds = qa_processor(kept_ds).materialize()

    final_ds = qa_ds.filter(
        lambda row: bool(row.get("qa_parse_ok"))
        and bool(_norm(row.get("qa_question")))
        and bool(_norm(row.get("qa_answer")))
    ).map(_to_output_record).materialize()

    final_count = final_ds.count()
    LOG.info("Accepted synthesized QA records: %d", final_count)

    final_ds.write_parquet(args.output_dir)
    LOG.info("Wrote synthesized QA parquet to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
