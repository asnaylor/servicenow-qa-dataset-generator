# Prompt Development

Tools for iterating on prompts and testing the LLM pipeline against individual tickets,
without running the full Ray pipeline.

---

## Standalone vLLM server

Start a local OpenAI-compatible vLLM server (no Ray) for interactive testing:

```bash
PODMANHPC_ADDITIONAL_STORES=/dvs_ro/cfs/cdirs/dasrepo/podman_shared_images \
podman-hpc run --rm -u 0 --gpu --nccl --net host --shm-size=40GB --group-add keep-groups \
  -v "${SCRATCH}/huggingface/:/home/ray/.cache/huggingface" \
  -v "${SCRATCH}/ray-llm-2.54.0-py311-libs/:/python_libs" \
  -e "HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)" \
  -e "PYTHONUSERBASE=/python_libs" \
  "rayproject/ray-llm:2.54.0-extra-py311-cu128" \
  vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key EMPTY \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
```

Minimal health-check request:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [
      {"role": "system", "content": "Return JSON only."},
      {"role": "user", "content": "Say hello in JSON with key message."}
    ],
    "max_tokens": 64,
    "temperature": 0.0
  }' | jq .
```

---

## Single-ticket test against a running vLLM server

`scripts/test_single_ticket_vllm_server.py` runs the full quality + QA prompt pair against
an already-running vLLM server.

1. Edit the constants at the top of the script:
   - `BASE_URL` — server address (e.g. `http://127.0.0.1:8000`)
   - `MODEL_NAME` — model ID served by vLLM
   - `INCIDENT_NUMBER` — ticket to test
   - `DOCS_INDEX` — path to the FAISS index prefix inside the container, or `None` to skip

2. Run:

```bash
PODMANHPC_ADDITIONAL_STORES=/dvs_ro/cfs/cdirs/dasrepo/podman_shared_images \
podman-hpc run --rm -u 0 --gpu --nccl --net host --shm-size=40GB --group-add keep-groups \
  -v "${SCRATCH}/huggingface/:/home/ray/.cache/huggingface" \
  -v "${PWD}:/workdir" \
  -v "${SCRATCH}/servicenow_incidents_jsonl:/tickets" \
  -v "${SCRATCH}/nersc_docs_index:/nersc_docs_index" \
  -v "${SCRATCH}/ray-llm-2.54.0-py311-libs/:/python_libs" \
  -e "HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || true)" \
  -e "PYTHONUSERBASE=/python_libs" \
  -w /workdir "rayproject/ray-llm:2.54.0-extra-py311-cu128" \
  python3.11 scripts/test_single_ticket_vllm_server.py
```
