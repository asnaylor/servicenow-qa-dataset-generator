# Pipeline

Three steps to go from raw ServiceNow JSON exports to a Parquet Q&A dataset.

## Step 1: Convert + Rule Filter

Converts one-ticket-per-file JSON exports into filtered JSONL shards.

```bash
python3.11 scripts/convert_servicenow_tickets_to_jsonl.py \
  --input-dir /path/to/servicenow_incidents \
  --output-dir ${SCRATCH}/servicenow_incidents_jsonl \
  --config config/qa_dataset.toml \
  --records-per-shard 10000 \
  --overwrite
```

Behavior:
- Applies rule-based filtering from `config/qa_dataset.toml` (state, close code, category, resource, date cutoff, assignment group)
- Writes accepted tickets as sharded JSONL (`*.jsonl`)
- Drops top-level `attachments` by default

Summary output printed on completion:
```
Wrote <N> ticket(s) into <M> shard(s)
Filtered out: <K>
Parse errors: <E>
```

Filter rules are documented in [configuration.md](configuration.md).

---

## Step 2: Index NERSC Documentation

Builds a FAISS vector index from the [nersc.gitlab.io](https://gitlab.com/NERSC/nersc.gitlab.io)
docs repo. Run once; reuse across pipeline runs. The index is used during Q&A synthesis to
retrieve relevant documentation sections and ground generated answers in official NERSC guidance.

Clone the docs repo:

```bash
git clone https://gitlab.com/NERSC/nersc.gitlab.io.git ${SCRATCH}/nersc.gitlab.io
```

Build the index:

```bash
PODMANHPC_ADDITIONAL_STORES=/dvs_ro/cfs/cdirs/dasrepo/podman_shared_images \
podman-hpc run --rm -u 0 --gpu --nccl --net host --shm-size=40GB --group-add keep-groups \
  -v "${SCRATCH}/huggingface/:/home/ray/.cache/huggingface" \
  -v "${PWD}:/workdir" \
  -v "${SCRATCH}/ray-llm-2.54.0-py311-libs/:/python_libs" \
  -e "HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)" \
  -e "PYTHONUSERBASE=/python_libs" \
  -v "${SCRATCH}/nersc.gitlab.io/:/nersc.gitlab.io" \
  -v "${SCRATCH}/nersc_docs_index/:/nersc_docs_index" \
  -w /workdir "rayproject/ray-llm:2.54.0-extra-py311-cu128" \
  python3.11 scripts/index_nersc_docs.py \
    --docs-root /nersc.gitlab.io/docs \
    --output /nersc_docs_index/index \
    --model BAAI/bge-large-en-v1.5
```

This produces two files inside `${SCRATCH}/nersc_docs_index/`:
- `index.faiss` — FAISS flat inner-product index
- `index.jsonl` — chunk metadata (path, URL, page title, section heading, text)

The `--model` flag must match `docs.model` in `config/llm_pipeline.toml`. Re-run whenever
the docs repo is updated.

Indexer options:
- `--model` — sentence-transformers model name (default: `BAAI/bge-small-en-v1.5`)
- `--max-chunk-chars` — maximum characters per chunk (default: `3000`)

---

## Step 3: LLM Quality Gate + Q&A Synthesis

Runs the two-stage LLM pipeline over the filtered JSONL shards.

```bash
python3.11 scripts/ray_servicenow_ticket_pipeline.py \
  --input ${SCRATCH}/servicenow_incidents_jsonl \
  --output-dir ${SCRATCH}/tickets_qa_parquet \
  --rejected-dir ${SCRATCH}/tickets_llm_rejected_parquet \
  --llm-config config/llm_pipeline.toml \
  --docs-index ${SCRATCH}/nersc_docs_index/index \
  --overwrite
```

**Stage 1 — Quality gate:** the LLM scores each ticket 0–100 for Q&A training usefulness.
Tickets scoring below the threshold (default 70) are written to `--rejected-dir` and skipped.

**Stage 2 — Q&A synthesis:** for each kept ticket, the LLM generates a structured JSON record
with `question`, `answer`, `summary`, and `tags`, grounded in relevant NERSC docs retrieved
from the index.

Output Parquet schema (accepted records):

| Column | Description |
|---|---|
| `incident_number` | Source ticket identifier |
| `sys_id` | ServiceNow sys_id |
| `source_path` | JSONL shard the ticket came from |
| `state` / `category` / `subcategory` / `u_resource` / `close_code` | Ticket metadata |
| `short_description` | Original ticket title (PII-redacted) |
| `qa_question` | Synthesized user question |
| `qa_answer` | Synthesized answer |
| `qa_summary` | One-sentence summary |
| `qa_tags` | List of technical tags |
| `quality_reason` | LLM's quality gate rationale |

Useful flags:
- `--ray-address auto` — connect to an existing Ray cluster
- `--limit-tickets N` — process only the first N tickets (smoke testing)
- `--docs-index` — path prefix to the FAISS index (omit to skip doc retrieval)
- `--overwrite` — replace existing output directories

LLM settings (model, concurrency, token limits, etc.) are documented in
[configuration.md](configuration.md).

For running at scale on SLURM, see [deployment.md](deployment.md).
