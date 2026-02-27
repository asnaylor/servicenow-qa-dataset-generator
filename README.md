# servicenow-qa-dataset-generator

Generate a Parquet Q&A dataset from ServiceNow exports using:
1) rule-based filtering during JSON -> JSONL conversion
2) LLM quality filtering
3) LLM Q&A synthesis

## Runtime

This project was developed and run inside the container image:

`rayproject/ray-llm:2.53.0-extra-py311-cu128`

If you run outside containers, use Python 3.11 with Ray Data + Ray Data LLM support.

## Pipeline Overview

Input expected by the LLM pipeline is **filtered JSONL shards** (`*.jsonl` or `*.jsonl.gz`) from the converter.

## Step 1: Convert + Rule Filter

If your source is one ticket per JSON file, run:

```bash
python3.11 scripts/convert_servicenow_tickets_to_jsonl.py \
  --input-dir /path/to/servicenow_incidents \
  --output-dir ${SCRATCH}/servicenow_incidents_jsonl \
  --config config/qa_dataset.toml \
  --records-per-shard 10000 \
  --overwrite
```

Converter behavior:
- Applies filtering rules from `config/qa_dataset.toml`
- Writes accepted tickets as sharded JSONL
- Tracks counts only (written / filtered / parse errors)
- Drops top-level `attachments` by default

Converter summary output:
- `Wrote <N> ticket(s) into <M> shard(s)`
- `Filtered out: <K>`
- `Parse errors: <E>`

## Step 2: LLM Quality + Q&A Synthesis

```bash
python3.11 scripts/ray_servicenow_ticket_pipeline.py \
  --input ${SCRATCH}/servicenow_incidents_jsonl \
  --output-dir ${SCRATCH}/tickets_qa_parquet \
  --rejected-dir ${SCRATCH}/tickets_llm_rejected_parquet \
  --llm-config config/llm_pipeline.toml \
  --overwrite
```

LLM pipeline behavior:
- Stage 1 quality gate: LLM decides `keep` / `reject` for QA usefulness
- Stage 2 synthesis: LLM generates JSON with `question`, `answer`, `summary`, `tags`
- Writes accepted synthesized Q&A records to Parquet (`--output-dir`)
- Optionally writes quality rejects to Parquet (`--rejected-dir`)

Useful knobs:
- `--ray-address auto` to run on an existing Ray cluster
- `--limit-tickets N` for smoke testing
- Edit `config/llm_pipeline.toml` for model source, concurrency, batch size, token limits, context size, and engine kwargs


## podman-hpc Example

If you run on a system where Ray/Python are provided via `podman-hpc`, mount:
- This repo into the container (so the scripts are available)
- JSONL input and output directories

```bash
PODMANHPC_ADDITIONAL_STORES=/dvs_ro/cfs/cdirs/dasrepo/podman_shared_images \
podman-hpc run --rm -u 0 --gpu --nccl --net host --shm-size=40GB --group-add keep-groups \
  -v "${SCRATCH}/huggingface/:/home/ray/.cache/huggingface" \
  -v "${PWD}:/workdir" \
  -v "${SCRATCH}/tickets_qa_parquet:/qa_parquet" \
  -v "${SCRATCH}/tickets_llm_rejected_parquet:/qa_rejected" \
  -v "${SCRATCH}/servicenow_incidents_jsonl:/tickets" \
  -e "HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)" \
  -w /workdir "rayproject/ray-llm:2.53.0-extra-py311-cu128" \
  python3.11 scripts/ray_servicenow_ticket_pipeline.py \
    --input /tickets \
    --output-dir /qa_parquet \
    --rejected-dir /qa_rejected \
    --llm-config config/llm_pipeline.toml \
    --limit-tickets 1000 \
    --overwrite
```

## SLURM (scale out with Ray)

This repo includes [scripts/deploy_ray_cluster.sh](scripts/deploy_ray_cluster.sh), which starts a Ray cluster across `N` SLURM nodes (head + workers) using `podman-hpc`, then runs the LLM pipeline on the Ray head.

1) Convert tickets to filtered JSONL once, and write them to a shared filesystem path visible from all nodes (for example `${SCRATCH}`). See [Step 1: Convert + Rule Filter](#step-1-convert--rule-filter).

2) Configure runtime paths and LLM config in `scripts/deploy_ray_cluster.sh` (or export env vars before `sbatch`):
- `TICKETS_JSONL_DIR`: input JSONL shard directory
- `KEPT_DIR`: output parquet directory for accepted synthesized Q&A
- `REJECTED_DIR`: output parquet directory for LLM-quality rejected tickets
- `LLM_CONFIG`: TOML settings file (default `config/llm_pipeline.toml`)

3) Submit with your desired node count, for example 4:

```bash
sbatch -N 4 scripts/deploy_ray_cluster.sh
```

Check `ray-cluster-<jobid>.log` for:
- Ray dashboard URL
- Cluster readiness
- Pipeline progress and final write locations

## Notes

- Rule-based filtering rules are defined in `config/qa_dataset.toml` (converter stage)
- LLM runtime/tuning settings are defined in `config/llm_pipeline.toml`
- LLM quality filtering and Q&A synthesis prompts are in `scripts/ray_servicenow_ticket_pipeline.py`
- Current pipeline writes Parquet outputs (not JSONL kept/rejected with `rejection_reasons`)
