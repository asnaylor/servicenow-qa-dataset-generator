# servicenow-qa-dataset-generator
Transform ServiceNow ticket JSON files into Q&amp;A training datasets using local LLMs. Filters low-quality tickets and extracts question-answer pairs at scale.

## Ray Data ticket filtering pipeline

The script `scripts/ray_servicenow_ticket_pipeline.py` reads ServiceNow ticket JSON files (one JSON object per file, like `81.json`), applies basic rejection rules (e.g. not-closed, auto-generated heuristics), and writes a normalized dataset.

### Install

```bash
pip install "ray[data]"
```

### Run

```bash
python scripts/ray_servicenow_ticket_pipeline.py \
  --input 81.json \
  --output-dir out/kept \
  --rejected-dir out/rejected \
  --output-format jsonl \
  --overwrite
```

### Run (podman-hpc)

If you run on a system where Ray is provided via `podman-hpc`, you must mount both:

- This repo into the container (so the script is available)
- Your ticket directory into the container (and then point `--input` at the *container* path)

Example (mount tickets at `/tickets`, then process `INC001`):

```bash
PODMANHPC_ADDITIONAL_STORES=/dvs_ro/cfs/cdirs/dasrepo/podman_shared_images \
podman-hpc run --rm -u 0 --gpu --nccl --net host --shm-size=40GB --group-add keep-groups \
  -v "${SCRATCH}/huggingface/:/home/ray/.cache/huggingface" \
  -v "${PWD}:/workdir" \
  -v "/dvs_ro/servicenow_incidents:/tickets" \
  -e "HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)" \
  -w /workdir "rayproject/ray-llm:2.53.0-extra-py311-cu128" \
  python scripts/ray_servicenow_ticket_pipeline.py \
    --input /tickets/INC001 \
    --output-dir out/kept \
    --rejected-dir out/rejected \
    --output-format jsonl \
    --overwrite
```

### Performance Options

```bash
# Control parallelism for better performance on large datasets
python scripts/ray_servicenow_ticket_pipeline.py \
  --input /path/to/tickets/*.json \
  --output-dir out/kept \
  --rejected-dir out/rejected \
  --output-format parquet \
  --concurrency 16 \
  --overwrite
```

### Notes

- Default kept tickets require `incident_fields.state == "Closed"` and an explicit `closed_at`.
- Auto-generated rejection is heuristic; use `--keep-auto-generated` to disable it.
- The pipeline uses Ray Data's `map_batches` for 2-5x better performance compared to row-by-row processing.
- Dataset is cached after normalization to avoid recomputing when splitting into kept/rejected outputs.
- Use `--concurrency N` to control the number of parallel read tasks (defaults to Ray auto-detection).
