# Configuration Reference

Two TOML files control the pipeline. Both live in `config/`.

---

## `config/qa_dataset.toml` — Rule-based ticket filter

Applied by `scripts/convert_servicenow_tickets_to_jsonl.py` (Step 1).
Tickets failing any rule are dropped before the LLM sees them.

### `[date]`

| Key | Description |
|---|---|
| `min_opened_date` | Reject tickets opened before this date (`YYYY-MM-DD`). Set to `""` to disable. Default `"2021-05-01"` (Perlmutter early access start). |

### `[state]`

| Key | Description |
|---|---|
| `allowed_states` | Only tickets in these states are kept. Default `["Closed", "Resolved"]`. |

### `[close_codes]`

| Key | Description |
|---|---|
| `allowed` | Keep tickets with these close codes. |
| `rejected` | Drop tickets with these close codes (evaluated after `allowed`). |

### `[contact_types]`

| Key | Description |
|---|---|
| `allowed` | Keep tickets submitted via these contact types. Default `["User Web Interface", "Email"]`. |
| `rejected` | Drop tickets with these contact types. |

### `[auto_generated]`

| Key | Description |
|---|---|
| `reject_if_short_description_matches` | List of regex patterns. Tickets whose `short_description` matches any pattern are dropped. |

### `[categories]`

| Key | Description |
|---|---|
| `allowed` | If non-empty, only these categories are kept. |
| `rejected` | Drop tickets in these categories regardless of `allowed`. |

### `[assignment_groups]`

| Key | Description |
|---|---|
| `rejected` | Drop tickets assigned to these groups (internal ops work, not user-facing support). |

### `[resources]`

| Key | Description |
|---|---|
| `allowed` | If non-empty, only these resources are kept. |
| `rejected` | Drop tickets for these resources (decommissioned systems: Cori, Edison, etc.). |

---

## `config/llm_pipeline.toml` — LLM pipeline settings

Applied by `scripts/ray_servicenow_ticket_pipeline.py` (Step 3).

### Top-level keys

| Key | Default | Description |
|---|---|---|
| `model_source` | `"meta-llama/Llama-3.1-8B-Instruct"` | HuggingFace model ID or local path passed to vLLM. |
| `batch_size` | `32` | Ray Data batch size per vLLM actor. |
| `context_max_chars` | `12000` | Maximum characters of ticket context fed to the LLM. Budget is split across metadata, problem description, resolution, comments, and work notes. |
| `concurrency` | `"auto"` | Number of vLLM actor replicas. `"auto"` resolves to `available_GPUs / tensor_parallel_size`. Set an integer to override. |

### `[engine_kwargs]`

Passed directly to vLLM's engine. Common keys:

| Key | Description |
|---|---|
| `tensor_parallel_size` | Number of GPUs per replica. |
| `max_model_len` | Maximum context window (tokens). |
| `gpu_memory_utilization` | Fraction of GPU VRAM reserved for the model (e.g. `0.90`). |
| `distributed_executor_backend` | **Must be `"mp"`** for multi-node Ray Data pipelines. Using the default `"ray"` backend causes vLLM to create Ray placement groups that outlive the streaming executor, blocking GPU resources between pipeline stages. |

### `[quality]`

Controls the quality gate (Stage 1).

| Key | Default | Description |
|---|---|---|
| `temperature` | `0.0` | Sampling temperature. `0.0` for deterministic scoring. |
| `max_tokens` | `128` | Maximum tokens in the quality decision JSON. |
| `score_threshold` | `70` | Tickets scoring below this (0–100) are rejected. |

### `[qa]`

Controls Q&A synthesis (Stage 2).

| Key | Default | Description |
|---|---|---|
| `temperature` | `0.1` | Sampling temperature. |
| `max_tokens` | `512` | Maximum tokens in the synthesized Q&A JSON. |

### `[docs]`

Controls NERSC documentation retrieval (used when `--docs-index` is passed).

| Key | Default | Description |
|---|---|---|
| `top_k` | `3` | Number of documentation chunks retrieved per ticket. |
| `model` | `"BAAI/bge-small-en-v1.5"` | Sentence-transformers embedding model. **Must match the model used when building the index** (`scripts/index_nersc_docs.py --model`). |
