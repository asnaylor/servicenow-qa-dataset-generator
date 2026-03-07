# servicenow-qa-dataset-generator

Generates a Parquet Q&A dataset from ServiceNow ticket exports using a three-stage pipeline:
rule-based filtering → LLM quality gate → LLM Q&A synthesis grounded in NERSC documentation.

Built on [Ray Data](https://docs.ray.io/en/latest/data/overview.html) + [vLLM](https://docs.vllm.ai),
designed to run inside `rayproject/ray-llm:2.54.0-extra-py311-cu128` on NERSC/Perlmutter.

## Pipeline at a glance

| Step | Script | Description |
|---|---|---|
| 1 | [`convert_servicenow_tickets_to_jsonl.py`](./scripts/convert_servicenow_tickets_to_jsonl.py) | Rule-based filter; JSON → JSONL shards |
| 2 | [`index_nersc_docs.py`](./scripts/index_nersc_docs.py) | Build FAISS index from NERSC docs repo |
| 3 | [`ray_servicenow_ticket_pipeline.py`](./scripts/ray_servicenow_ticket_pipeline.py) | LLM quality gate + Q&A synthesis → Parquet |

## Quick start

```bash
# Step 1: filter tickets
python3.11 scripts/convert_servicenow_tickets_to_jsonl.py \
  --input-dir /path/to/servicenow_incidents \
  --output-dir ${SCRATCH}/servicenow_incidents_jsonl \
  --config config/qa_dataset.toml --overwrite

# Step 2: index NERSC docs (once)
python3.11 scripts/index_nersc_docs.py \
  --docs-root ${SCRATCH}/nersc.gitlab.io/docs \
  --output ${SCRATCH}/nersc_docs_index/index \
  --model BAAI/bge-large-en-v1.5

# Step 3: run LLM pipeline (single node)
python3.11 scripts/ray_servicenow_ticket_pipeline.py \
  --input ${SCRATCH}/servicenow_incidents_jsonl \
  --output-dir ${SCRATCH}/tickets_qa_parquet \
  --rejected-dir ${SCRATCH}/tickets_llm_rejected_parquet \
  --llm-config config/llm_pipeline.toml \
  --docs-index ${SCRATCH}/nersc_docs_index/index \
  --overwrite
```

For SLURM/multi-node runs: `sbatch -N 4 scripts/deploy_ray_cluster.sh`

## Runtime

Container image: `rayproject/ray-llm:2.54.0-extra-py311-cu128`

`faiss-cpu` and `sentence-transformers` are not included in the image. Install once into
a persistent scratch directory and mount it on every container run via `PYTHONUSERBASE`:

```bash
# Inside the container (with /python_libs mounted):
pip install --user faiss-cpu sentence-transformers
```

All `podman-hpc` commands in the docs include the required mounts:
```
-v "${SCRATCH}/ray-llm-2.54.0-py311-libs/:/python_libs"
-e "PYTHONUSERBASE=/python_libs"
```

## Documentation

| Doc | Description |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | Step-by-step pipeline walkthrough with full commands |
| [docs/configuration.md](docs/configuration.md) | Reference for `qa_dataset.toml` and `llm_pipeline.toml` |
| [docs/deployment.md](docs/deployment.md) | SLURM/Ray cluster setup and manual `podman-hpc` invocation |
| [docs/prompt-development.md](docs/prompt-development.md) | Single-ticket testing, standalone vLLM server, prompt iteration |

## Notebooks

| Notebook | Description |
|---|---|
| [`notebooks/read_llm_outputs.ipynb`](notebooks/read_llm_outputs.ipynb) | Explore and inspect the Parquet Q&A output from the pipeline |