# servicenow-qa-dataset-generator

Transform ServiceNow ticket JSON files into normalized datasets (and downstream Q&A training data). The core pipeline uses Ray Data to filter low-quality tickets (e.g., not closed; likely auto-generated) and write kept/rejected datasets.

## Runtime

This project was developed and run inside the container image:

`rayproject/ray-llm:2.53.0-extra-py311-cu128`

If you run outside containers, use Python 3.11 with Ray Data available.

## Input Format

The Ray pipeline reads **JSONL shards** (one ticket per line, `*.jsonl` or `*.jsonl.gz`).

If your tickets are exported as **one JSON object per file** (often with large `attachments`), convert them to sharded JSONL first. The converter drops the top-level `attachments` field by default.

```bash
python3.11 scripts/convert_servicenow_tickets_to_jsonl.py \
  --input-dir /path/to/servicenow_incidents \
  --output-dir ${SCRATCH}/servicenow_incidents_jsonl \
  --records-per-shard 10000 \
  --overwrite
```

## Run Pipeline

```bash
python3.11 scripts/ray_servicenow_ticket_pipeline.py \
  --input ${SCRATCH}/servicenow_incidents_jsonl \
  --output-dir ${SCRATCH}/tickets_kept \
  --rejected-dir ${SCRATCH}/tickets_rejected \
  --output-format jsonl \
  --overwrite
```

## Browse Results

For quick one-line summaries and drill-down viewing of kept/rejected outputs:

```bash
python scripts/ticket_browser.py \
  --kept ${SCRATCH}/tickets_kept \
  --rejected ${SCRATCH}/tickets_rejected
```

## podman-hpc

If you run on a system where Ray/Python are provided via `podman-hpc`, mount:

- This repo into the container (so the scripts are available)
- Your ticket directory into the container

Example (mount tickets at `/tickets`, convert `INC001` to JSONL, then run the pipeline):

```bash
PODMANHPC_ADDITIONAL_STORES=/dvs_ro/cfs/cdirs/dasrepo/podman_shared_images \
podman-hpc run --rm -u 0 --gpu --nccl --net host --shm-size=40GB --group-add keep-groups \
  -v "${SCRATCH}/huggingface/:/home/ray/.cache/huggingface" \
  -v "${PWD}:/workdir" \
  -v "${SCRATCH}/tickets_kept:/kept" \
  -v "${SCRATCH}/tickets_rejected:/rejected" \
  -v "${SCRATCH}/servicenow_incidents_jsonl:/tickets" \
  -e "HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)" \
  -w /workdir "rayproject/ray-llm:2.53.0-extra-py311-cu128" \
  python3.11 scripts/ray_servicenow_ticket_pipeline.py \
    --input /tickets \
    --output-dir /kept \
    --rejected-dir /rejected \
    --output-format jsonl \
    --limit-files 1000 \
    --overwrite
```

## SLURM (scale out with Ray)

This repo includes a SLURM script that starts a Ray cluster across `N` nodes (head + workers) using `podman-hpc`, then runs the ticket pipeline on the Ray head.

1) Convert tickets to JSONL (once), writing to a shared filesystem path visible from all nodes (e.g. `${SCRATCH}`):

```bash
python3.11 scripts/convert_servicenow_tickets_to_jsonl.py \
  --input-dir /path/to/servicenow_incidents \
  --output-dir ${SCRATCH}/servicenow_incidents_jsonl \
  --records-per-shard 10000 \
  --overwrite
```

2) Edit the path variables at the top of `scripts/deploy_ray_cluster.sh` (or export them before `sbatch`) to point to your JSONL + output dirs:

- `TICKETS_JSONL_DIR` (input JSONL shards)
- `KEPT_DIR` and `REJECTED_DIR` (outputs)

3) Submit the job with the desired node count, e.g. 4 nodes:

```bash
sbatch -N 4 scripts/deploy_ray_cluster.sh
```

Check logs in `ray-cluster-<jobid>.log` for the Ray dashboard URL and pipeline progress.

## Notes

- Default kept tickets require `incident_fields.state == "Closed"` and an explicit `closed_at`.
- Auto-generated rejection is heuristic; use `--keep-auto-generated` to disable it.
- Use `--read-num-blocks N` to control JSONL read parallelism (Ray 2.53.0 uses `override_num_blocks`).
