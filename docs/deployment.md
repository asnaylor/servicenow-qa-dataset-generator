# Deployment

## SLURM + Ray cluster (recommended for full runs)

`scripts/deploy_ray_cluster.sh` starts a Ray cluster across N SLURM nodes (head + workers)
using `podman-hpc`, then runs the LLM pipeline on the head node.

### Prerequisites

1. Filtered JSONL shards written to a shared filesystem path (see [pipeline.md Step 1](pipeline.md)).
2. NERSC docs index built and on a shared filesystem (see [pipeline.md Step 2](pipeline.md)).
3. Extra Python packages (`faiss-cpu`, `sentence-transformers`) installed into a persistent
   scratch directory and available via `PYTHONUSERBASE` (see the Runtime section in the README).

### Configuration

Override any of these environment variables before calling `sbatch`:

| Variable | Default | Description |
|---|---|---|
| `TICKETS_JSONL_DIR` | `${SCRATCH}/servicenow_incidents_jsonl` | Input JSONL shard directory (shared filesystem). |
| `KEPT_DIR` | `${SCRATCH}/servicenow_incidents_processed/qa_parquet` | Output Parquet for accepted Q&A records. |
| `REJECTED_DIR` | `${SCRATCH}/servicenow_incidents_processed/llm_rejected_parquet` | Output Parquet for quality-rejected tickets. |
| `DOCS_INDEX_DIR` | `${SCRATCH}/nersc_docs_index` | Directory containing `index.faiss` and `index.jsonl`. Set to `""` to skip doc retrieval. |
| `LLM_CONFIG` | `config/llm_pipeline.toml` | Path to the LLM TOML config (relative to repo root). |
| `LIMIT_TICKETS` | _(unset)_ | Set to an integer to cap the number of tickets processed. |
| `PYTHON_LIBS_DIR` | `${SCRATCH}/ray-llm-2.54.0-py311-libs` | Directory containing extra pip packages. |

### Submitting

The script defaults to 2 nodes. Override with `-N` for larger runs:

```bash
sbatch -N 4 scripts/deploy_ray_cluster.sh
```

With `tensor_parallel_size = 4` (4 GPUs per replica) and `concurrency = "auto"`:
- 2 nodes × 4 GPUs = 2 replicas
- 4 nodes × 4 GPUs = 4 replicas

Check `ray-cluster-<jobid>.log` for the Ray dashboard URL, cluster readiness, and
pipeline progress.

### How it works

1. Starts the Ray head node on the first allocated node via `srun`.
2. Polls the Ray dashboard REST API until the head is alive.
3. Starts Ray worker nodes on remaining nodes via `srun`.
4. Polls the dashboard until all nodes are connected.
5. Runs `ray_servicenow_ticket_pipeline.py` inside a driver container on the head node.

---

## Manual podman-hpc invocation (single node or testing)

```bash
PODMANHPC_ADDITIONAL_STORES=/dvs_ro/cfs/cdirs/dasrepo/podman_shared_images \
podman-hpc run --rm -u 0 --gpu --nccl --net host --shm-size=40GB --group-add keep-groups \
  -v "${SCRATCH}/huggingface/:/home/ray/.cache/huggingface" \
  -v "${PWD}:/workdir" \
  -v "${SCRATCH}/servicenow_incidents_jsonl:/tickets" \
  -v "${SCRATCH}/tickets_qa_parquet:/qa_parquet" \
  -v "${SCRATCH}/tickets_llm_rejected_parquet:/qa_rejected" \
  -v "${SCRATCH}/nersc_docs_index:/nersc_docs_index" \
  -v "${SCRATCH}/ray-llm-2.54.0-py311-libs/:/python_libs" \
  -e "HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)" \
  -e "PYTHONUSERBASE=/python_libs" \
  -w /workdir "rayproject/ray-llm:2.54.0-extra-py311-cu128" \
  python3.11 scripts/ray_servicenow_ticket_pipeline.py \
    --input /tickets \
    --output-dir /qa_parquet \
    --rejected-dir /qa_rejected \
    --llm-config config/llm_pipeline.toml \
    --docs-index /nersc_docs_index/index \
    --overwrite
```

Add `--limit-tickets 50` for a quick smoke test.

---

## Container image

```
rayproject/ray-llm:2.54.0-extra-py311-cu128
```

If running outside containers, use Python 3.11 with Ray 2.54.0+ and Ray Data LLM support.
