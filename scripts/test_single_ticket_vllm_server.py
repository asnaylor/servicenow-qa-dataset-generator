#!/usr/bin/env python3.11
"""
Single-ticket prompt test against a running vLLM OpenAI-compatible server.

Edit the constants below, then run:
  python3.11 scripts/test_single_ticket_vllm_server.py
"""

import gzip
import json
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

import ray_servicenow_ticket_pipeline as pipeline

# ----------------------------- edit these -----------------------------
BASE_URL = "http://nid001052:8000"
API_KEY = "EMPTY"
MODEL_NAME = "openai/gpt-oss-120b"

INPUT_PATHS = ["/tickets"]
INCIDENT_NUMBER = "INC0237766" #INC0163628, INC0215466, INC0237766  - medium tickets INC0196540 - poor tickets INC0124780
LLM_CONFIG_PATH = "config/llm_pipeline.toml"
DOCS_INDEX = "/nersc_docs_index/index"  # set to None to skip docs retrieval

FORCE_QA_IF_REJECTED = False
# ---------------------------------------------------------------------


def _open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _incident_number_from_ticket(ticket: Mapping[str, Any]) -> str:
    incident = ticket.get("incident_fields") or {}
    metadata = ticket.get("metadata") or {}
    if not isinstance(incident, dict):
        incident = {}
    if not isinstance(metadata, dict):
        metadata = {}
    value = incident.get("number") or metadata.get("incident_number") or ticket.get("incident_number")
    return pipeline._norm(value)  # pylint: disable=protected-access


def _iter_tickets(input_files: Iterable[str]) -> Iterable[dict[str, Any]]:
    for raw_path in input_files:
        path = Path(raw_path)
        with _open_jsonl(path) as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj


def _find_ticket_by_incident(input_files: list[str], incident_number: str) -> dict[str, Any] | None:
    needle = pipeline._norm(incident_number)  # pylint: disable=protected-access
    for ticket in _iter_tickets(input_files):
        if _incident_number_from_ticket(ticket) == needle:
            return ticket
    return None


def _chat_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    sampling_params: Mapping[str, Any],
    structured_outputs: Mapping[str, Any] | None,
    schema_name: str | None = None,
) -> str:
    url = f"{BASE_URL.rstrip('/')}/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(sampling_params.get("temperature", 0.0)),
        "max_tokens": int(sampling_params.get("max_tokens", 256)),
    }
    if structured_outputs:
        payload["structured_outputs"] = dict(structured_outputs)
        # Per vLLM docs (online serving OpenAI API), JSON schema is supported via response_format.
        # See: https://docs.vllm.ai/en/latest/features/structured_outputs/#online-serving-openai-api
        json_schema = structured_outputs.get("json") if isinstance(structured_outputs, Mapping) else None
        if isinstance(json_schema, Mapping) and json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name or "structured-output",
                    "schema": dict(json_schema),
                },
            }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    return (
        obj.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )


def _structured_outputs_from_sampling_params(sampling_params: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """
    Extract schema-guidance config from Ray-style sampling_params.

    Ray uses a version-dependent key:
    - "structured_outputs" (newer)
    - "guided_decoding" (older)

    vLLM's OpenAI-compatible server expects this at the top level as "structured_outputs".
    """
    if not isinstance(sampling_params, Mapping):
        return None
    cfg = sampling_params.get("structured_outputs")
    if isinstance(cfg, Mapping) and cfg:
        return cfg
    cfg = sampling_params.get("guided_decoding")
    if isinstance(cfg, Mapping) and cfg:
        return cfg
    return None


def _print_section(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def main() -> int:
    llm_cfg = pipeline._load_toml_config(LLM_CONFIG_PATH)  # pylint: disable=protected-access
    context_max_chars = int(llm_cfg.get("context_max_chars", 12000))

    quality_cfg = llm_cfg.get("quality", {})
    if not isinstance(quality_cfg, dict):
        quality_cfg = {}
    qa_cfg = llm_cfg.get("qa", {})
    if not isinstance(qa_cfg, dict):
        qa_cfg = {}

    input_files = pipeline._discover_local_files(  # pylint: disable=protected-access
        INPUT_PATHS,
        recursive=True,
        allowed_suffixes=(".jsonl", ".jsonl.gz"),
    )
    ticket = _find_ticket_by_incident(input_files, INCIDENT_NUMBER)
    if ticket is None:
        raise SystemExit(f"Incident not found: {INCIDENT_NUMBER}")

    input_record = pipeline._to_llm_input_record(  # pylint: disable=protected-access
        ticket, context_max_chars=context_max_chars
    )

    docs_cfg = llm_cfg.get("docs", {})
    if not isinstance(docs_cfg, dict):
        docs_cfg = {}
    if DOCS_INDEX:
        retriever = pipeline._build_docs_retriever(  # pylint: disable=protected-access
            DOCS_INDEX,
            top_k=int(docs_cfg.get("top_k", 3)),
            model_name=str(docs_cfg.get("model", "BAAI/bge-large-en-v1.5")),
        )
        input_record = retriever(input_record)
    else:
        input_record = {**input_record, "docs_context": "", "docs_chunks": []}

    _print_section("RETRIEVED DOCS CHUNKS")
    docs_chunks = input_record.get("docs_chunks") or []
    if docs_chunks:
        for i, heading in enumerate(docs_chunks, start=1):
            print(f"  {i}. {heading}")
    else:
        print("  (none — docs index not loaded or no short_description)")

    quality_pre = pipeline._build_quality_preprocess(  # pylint: disable=protected-access
        temperature=float(quality_cfg.get("temperature", 0.0)),
        max_tokens=int(quality_cfg.get("max_tokens", 128)),
        context_chars=context_max_chars,
    )
    qa_pre = pipeline._build_qa_preprocess(  # pylint: disable=protected-access
        temperature=float(qa_cfg.get("temperature", 0.1)),
        max_tokens=int(qa_cfg.get("max_tokens", 512)),
        context_chars=context_max_chars,
    )

    quality_req = quality_pre(input_record)
    _print_section("QUALITY PROMPT")
    print(json.dumps(quality_req.get("messages"), indent=2))

    quality_text = _chat_completion(
        model=MODEL_NAME,
        messages=quality_req.get("messages") or [],
        sampling_params=quality_req.get("sampling_params") or {},
        structured_outputs=_structured_outputs_from_sampling_params(quality_req.get("sampling_params") or {}),
        schema_name="quality-decision",
    )
    quality_score_threshold = int(quality_cfg.get("score_threshold", 70))
    quality_postprocess = pipeline._build_quality_postprocess(  # pylint: disable=protected-access
        score_threshold=quality_score_threshold
    )
    quality_row = quality_postprocess({**quality_req, "generated_text": quality_text})

    _print_section("QUALITY OUTPUT")
    print("quality_score:", quality_row.get("quality_score", 0))
    print("quality_keep:", bool(quality_row.get("quality_keep")))
    print("quality_reason:", pipeline._norm(quality_row.get("quality_reason")))  # pylint: disable=protected-access
    print("quality_raw:")
    print(pipeline._norm(quality_row.get("quality_raw")))  # pylint: disable=protected-access

    if not bool(quality_row.get("quality_keep")) and not FORCE_QA_IF_REJECTED:
        _print_section("QA SKIPPED")
        print("Set FORCE_QA_IF_REJECTED = True to run QA anyway.")
        return 0

    # Carry docs_context forward — the quality preprocess doesn't pass it through.
    quality_row = {**quality_row, "docs_context": input_record.get("docs_context", "")}

    qa_input = qa_pre(quality_row)
    _print_section("QA PROMPT")
    print(json.dumps(qa_input.get("messages"), indent=2))

    # Debug: show actual max_tokens being used
    actual_max_tokens = qa_input.get("sampling_params", {}).get("max_tokens", "NOT SET")
    print(f"\n[DEBUG] max_tokens being sent to vLLM: {actual_max_tokens}")

    qa_text = _chat_completion(
        model=MODEL_NAME,
        messages=qa_input.get("messages") or [],
        sampling_params=qa_input.get("sampling_params") or {},
        structured_outputs=_structured_outputs_from_sampling_params(qa_input.get("sampling_params") or {}),
        schema_name="qa-synthesis",
    )
    qa_row = pipeline._qa_postprocess(  # pylint: disable=protected-access
        {**qa_input, "generated_text": qa_text}
    )

    _print_section("QA OUTPUT")
    print("qa_parse_ok:", bool(qa_row.get("qa_parse_ok")))
    print("qa_question:", pipeline._norm(qa_row.get("qa_question")))  # pylint: disable=protected-access
    print("qa_answer:", pipeline._norm(qa_row.get("qa_answer")))  # pylint: disable=protected-access
    print("qa_summary:", pipeline._norm(qa_row.get("qa_summary")))  # pylint: disable=protected-access
    print("qa_tags:", qa_row.get("qa_tags") or [])
    print("qa_raw:")
    print(pipeline._norm(qa_row.get("qa_raw")))  # pylint: disable=protected-access
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
