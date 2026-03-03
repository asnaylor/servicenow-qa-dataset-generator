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

from packaging.version import Version
from pydantic import BaseModel, Field
import tomllib

LOG = logging.getLogger("ray_servicenow_ticket_pipeline")


class QualityDecision(BaseModel):
    score: int = Field(ge=0, le=100, description="Overall quality score 0-100")
    keep: bool
    reason: str = Field(min_length=1, max_length=200)


class QaSynthesis(BaseModel):
    question: str = Field(min_length=1, max_length=800)
    answer: str = Field(min_length=1, max_length=6400)
    summary: str = Field(min_length=1, max_length=400)
    tags: list[str] = Field(min_length=1, max_length=8)


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _model_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    schema = model_cls.model_json_schema()
    if not isinstance(schema, dict):
        raise TypeError(f"Schema generation failed for model: {model_cls}")
    return schema


QUALITY_JSON_SCHEMA = _model_json_schema(QualityDecision)
QA_JSON_SCHEMA = _model_json_schema(QaSynthesis)


def _get_structured_outputs_key() -> str:
    """
    Return the correct sampling_params key for structured outputs based on Ray version.

    Ray 2.53.0 and earlier use "guided_decoding" (deprecated).
    Ray 2.54.0 and later use "structured_outputs" (current).

    See:
    - Ray 2.53.0: https://github.com/ray-project/ray/blob/ray-2.53.0/python/ray/llm/_internal/batch/stages/vllm_engine_stage.py
    - Ray 2.54.0+: https://github.com/ray-project/ray/blob/master/python/ray/llm/_internal/batch/stages/vllm_engine_stage.py
    """
    try:
        import ray
        ray_version = Version(ray.__version__)
        # Ray 2.54.0+ uses "structured_outputs"
        if ray_version >= Version("2.54.0"):
            return "structured_outputs"
        # Ray 2.53.0 and earlier use "guided_decoding"
        return "guided_decoding"
    except Exception:
        # Default to newer API if version detection fails
        return "structured_outputs"


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


def _serialize_ticket_json(ticket: Mapping[str, Any]) -> str:
    try:
        return json.dumps(ticket, sort_keys=True, separators=(",", ":"))
    except Exception:
        return "{}"


def _smart_truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at sentence boundary, not mid-sentence."""
    if len(text) <= max_chars:
        return text

    # Try to find last sentence boundary before max_chars
    truncated = text[:max_chars]
    for delimiter in [". ", ".\n", "! ", "?\n", "? "]:
        last_delim = truncated.rfind(delimiter)
        if last_delim > max_chars * 0.7:  # At least 70% of budget used
            return truncated[:last_delim + 1].rstrip()

    # Fallback: try to break at newline
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.7:
        return truncated[:last_newline].rstrip()

    # Last resort: hard truncate with ellipsis
    return truncated.rstrip() + "..."


def _extract_comments_smart(entries: Any, budget_chars: int, prioritize_resolution: bool = True) -> str:
    """Extract comments with smart truncation, prioritizing resolution info."""
    if not isinstance(entries, list) or not entries:
        return ""

    # Extract and clean all comments
    comments = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = _norm(entry.get("text"))
        if text:
            comments.append(text)

    if not comments:
        return ""

    # If prioritizing resolution, reverse to get latest (resolution) comments first
    if prioritize_resolution and len(comments) > 1:
        comments = list(reversed(comments))

    # Build output staying within budget
    result = []
    used_chars = 0

    for i, comment in enumerate(comments):
        # Reserve space for "...[X more]" if needed
        remaining = budget_chars - used_chars
        if i < len(comments) - 1:
            remaining -= 20  # Reserve for truncation message

        if remaining <= 50:  # Not enough space
            omitted = len(comments) - i
            if omitted > 0:
                result.append(f"...[{omitted} more comment(s) omitted]")
            break

        # Truncate this comment if needed
        if len(comment) > remaining:
            comment = _smart_truncate_at_sentence(comment, remaining)

        result.append(comment)
        used_chars += len(comment)

    # Restore original order if we reversed
    if prioritize_resolution and len(result) > 1 and not result[-1].startswith("...["):
        result = list(reversed(result))

    return "\n\n".join(result)


def _ticket_to_context(ticket: Mapping[str, Any], *, max_chars: int) -> str:
    """
    Smart ticket context extraction with hierarchical token budgeting.

    Token budget allocation (% of max_chars):
    - Metadata (category, resource, etc.):  10%
    - Problem description:                  15%
    - Resolution (close notes):             25%
    - Customer comments:                    25%
    - Internal notes:                       25%
    """
    incident = ticket.get("incident_fields") or {}
    discussions = ticket.get("discussions") or {}
    metadata = ticket.get("metadata") or {}

    if not isinstance(incident, dict):
        incident = {}
    if not isinstance(discussions, dict):
        discussions = {}
    if not isinstance(metadata, dict):
        metadata = {}

    # Allocate character budgets
    budget_metadata = int(max_chars * 0.10)
    budget_problem = int(max_chars * 0.15)
    budget_resolution = int(max_chars * 0.25)
    budget_comments = int(max_chars * 0.25)
    budget_notes = int(max_chars * 0.25)

    # Build compact metadata section
    resource = _norm(incident.get("u_resource"))
    category = _norm(incident.get("category"))
    subcategory = _norm(incident.get("subcategory"))
    close_code = _norm(incident.get("close_code"))

    metadata_parts = []
    if resource:
        metadata_parts.append(f"Resource: {resource}")
    if category:
        cat_str = category
        if subcategory:
            cat_str += f" > {subcategory}"
        metadata_parts.append(f"Category: {cat_str}")
    if close_code:
        metadata_parts.append(f"Resolved: {close_code}")

    metadata_section = " | ".join(metadata_parts)
    if len(metadata_section) > budget_metadata:
        metadata_section = _smart_truncate_at_sentence(metadata_section, budget_metadata)

    # Problem description (short desc + description field)
    short_desc = _norm(incident.get("short_description"))
    description = _norm(incident.get("description"))

    problem_section = ""
    if short_desc:
        problem_section = f"Issue: {short_desc}"
        if description and len(problem_section) < budget_problem:
            remaining = budget_problem - len(problem_section) - 10
            if remaining > 50:
                desc_truncated = _smart_truncate_at_sentence(description, remaining)
                if desc_truncated:
                    problem_section += f"\nDetails: {desc_truncated}"
    elif description:
        problem_section = _smart_truncate_at_sentence(description, budget_problem)

    # Resolution info (close notes) - HIGH PRIORITY
    close_notes = _norm(incident.get("close_notes"))
    resolution_section = ""
    if close_notes:
        resolution_section = _smart_truncate_at_sentence(close_notes, budget_resolution)

    # Customer comments - prioritize resolution (latest) comments
    comments_section = _extract_comments_smart(
        discussions.get("customer_facing_comments"),
        budget_comments,
        prioritize_resolution=True
    )

    # Internal work notes - prioritize resolution
    notes_section = _extract_comments_smart(
        discussions.get("internal_work_notes"),
        budget_notes,
        prioritize_resolution=True
    )

    # Assemble final context with compact formatting
    sections = []

    if metadata_section:
        sections.append(metadata_section)

    if problem_section:
        sections.append(f"\n{problem_section}")

    if resolution_section:
        sections.append(f"\nResolution:\n{resolution_section}")

    if comments_section:
        sections.append(f"\nComments:\n{comments_section}")

    if notes_section:
        sections.append(f"\nWork Notes:\n{notes_section}")

    context = "\n".join(sections)

    # Final safety check (should rarely trigger now)
    if len(context) > max_chars:
        context = _smart_truncate_at_sentence(context, max_chars)

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
        "original_ticket_json": _serialize_ticket_json(ticket),
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    s = (text or "").strip()
    if not s:
        return None

    # Common wrappers: fenced blocks, leading "json", etc.
    if "```" in s:
        s = s.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    if s.lower().startswith("json"):
        # Some models return "json\n{...}".
        s = s[3:].lstrip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Lenient extraction: find a JSON object substring inside larger text.
    start = s.find("{")
    if start < 0:
        return None

    # Try progressively shorter suffixes ending at a '}' to handle trailing chatter.
    # If the model output is truncated and missing a closing brace, this will return None.
    for end in range(len(s) - 1, start, -1):
        if s[end] != "}":
            continue
        candidate = s[start : end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue
    return None


def _build_quality_preprocess(
    *,
    temperature: float,
    max_tokens: int,
    context_chars: int,
):
    system_prompt = (
        "You are an expert QA dataset curator for technical IT support tickets. "
        "Score each ticket 0-100 based on its usefulness for creating question-answer training pairs."
    )

    # Determine which key to use for structured outputs based on Ray version
    structured_key = _get_structured_outputs_key()
    structured_cfg = {"json": QUALITY_JSON_SCHEMA}

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        context = _norm(row.get("ticket_context"))[:context_chars]
        user_prompt = (
            "Score this ticket 0-100 for QA training usefulness. "
            'Return ONLY JSON: {"score": 0-100, "keep": true/false, "reason": "brief explanation"}.\n\n'

            "Scoring (total 100):\n"
            "• Problem clarity (30): specific errors/versions vs vague \"doesn't work\"\n"
            "• Solution completeness (40): detailed commands/steps vs \"user resolved\"\n"
            "• Technical depth (20): specific configs/versions vs generic advice\n"
            "• Reproducibility (10): can others follow vs too vague\n\n"

            "Examples: 85+=detailed error+commands, 70-84=clear problem+steps, "
            "50-69=generic fix, <50=vague/admin. Keep if score>=70. Be strict.\n\n"

            "IMPORTANT: Keep reason under 20 words. Just state key strengths/weaknesses.\n\n"

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
            "original_ticket_json": _norm(row.get("original_ticket_json")),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "sampling_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                structured_key: {
                    **structured_cfg,
                },
            },
        }

    return preprocess


def _build_quality_postprocess(*, score_threshold: int = 70):
    """Build quality postprocess function with score threshold."""
    def postprocess(row: dict[str, Any]) -> dict[str, Any]:
        raw = _norm(row.get("generated_text"))
        parsed = _extract_json_object(raw)

        score = 0
        keep = False
        reason = "invalid_json_from_llm"

        if parsed is not None:
            score = int(parsed.get("score", 0))
            # Validate score range
            score = max(0, min(100, score))
            # Override keep based on threshold (don't trust LLM's keep value)
            keep = score >= score_threshold
            reason = _norm(parsed.get("reason")) or f"score_{score}"
        elif raw.lstrip().startswith("{"):
            reason = "invalid_json_truncated"
            # Best-effort salvage of score when JSON is cut off
            m = re.search(r"\"score\"\\s*:\\s*(\d+)", raw, flags=re.IGNORECASE)
            if m:
                score = int(m.group(1))
                score = max(0, min(100, score))
                keep = score >= score_threshold

        return {
            **row,
            "quality_score": score,
            "quality_keep": keep,
            "quality_reason": reason,
            "quality_raw": raw,
        }

    return postprocess


def _build_qa_preprocess(
    *,
    temperature: float,
    max_tokens: int,
    context_chars: int,
):
    system_prompt = (
        "You are an expert technical writer. Convert a ServiceNow incident ticket "
        "into a concise synthetic Q&A pair for training. "
        "Output must follow the JSON schema."
    )

    # Determine which key to use for structured outputs based on Ray version
    structured_key = _get_structured_outputs_key()
    structured_cfg = {"json": QA_JSON_SCHEMA}

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        context = _norm(row.get("ticket_context"))[:context_chars]
        user_prompt = (
            "Return ONLY valid JSON with keys:\n"
            "- question (string)\n"
            "- answer (string)\n"
            "- summary (string)\n"
            "- tags (array of strings)\n"
            "The answer must be grounded in the ticket details.\n\n"
            "Constraints:\n"
            "- Output MUST be a single JSON object (no markdown, no code fences, no extra text).\n"
            "- Keep question detailed and specific (under 800 characters).\n"
            "- Keep summary comprehensive (under 400 characters).\n"
            "- Keep answer detailed (under 6400 characters) with concrete steps, commands, and explanations.\n"
            "- tags: 1-8 short strings.\n\n"
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
            "original_ticket_json": _norm(row.get("original_ticket_json")),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "sampling_params": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                structured_key: {
                    **structured_cfg,
                },
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
        "original_ticket_json": _norm(row.get("original_ticket_json")),
    }


def _load_toml_config(config_path: str) -> dict[str, Any]:
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
    quality_score_threshold = int(quality_cfg.get("score_threshold", 70))

    qa_cfg = llm_cfg.get("qa", {})
    if not isinstance(qa_cfg, dict):
        qa_cfg = {}
    qa_temperature = float(qa_cfg.get("temperature", 0.1))
    qa_max_tokens = int(qa_cfg.get("max_tokens", 512))

    base_engine_kwargs = llm_cfg.get("engine_kwargs", {})
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

    structured_key = _get_structured_outputs_key()
    LOG.info("Using structured outputs key: %s (Ray version-aware)", structured_key)
    LOG.info("Using vLLM engine kwargs: %s", base_engine_kwargs)
    LOG.info("Using LLM concurrency=%d", concurrency)
    LOG.info("Using quality score threshold=%d", quality_score_threshold)
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
        postprocess=_build_quality_postprocess(score_threshold=quality_score_threshold),
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
                "quality_score": row.get("quality_score", 0),
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
