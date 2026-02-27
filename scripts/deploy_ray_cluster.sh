#!/bin/bash
#SBATCH --job-name=ray-cluster
#SBATCH --account=nstaff
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=ray-cluster-%j.log

# Bash best practices
set -euo pipefail  

# Configuration
export RAY_PORT=6379
export RAY_DASHBOARD_PORT=8265
export RAY_TEMP_DIR="${SCRATCH}/tmp/ray/${SLURM_JOB_ID}"
mkdir -p ${RAY_TEMP_DIR}
HEAD_NODE_TIMEOUT=300  # seconds to wait for head node
WORKER_TIMEOUT=300  # seconds to wait for all workers to connect
export PODMANHPC_ADDITIONAL_STORES=/dvs_ro/cfs/cdirs/dasrepo/podman_shared_images
export RAY_IMAGE="rayproject/ray-llm:2.53.0-extra-py311-cu128"
export HF_DIR="${HF_HOME:-${SCRATCH}/huggingface/}"
HF_TOKEN="$(cat ~/.hf_token 2>/dev/null || true)"
export HF_KEY="${HF_TOKEN:-}"

#------------------------------------------------------------------------------
# ServiceNow pipeline paths (edit these)
#
# Ray tasks execute inside the containers started by `ray start` on each node.
# Any input/output paths used by the pipeline MUST be mounted into ALL Ray
# containers (head + workers + the driver container).
#------------------------------------------------------------------------------

# Directory containing JSONL shards (*.jsonl or *.jsonl.gz).
TICKETS_JSONL_DIR="${TICKETS_JSONL_DIR:-${SCRATCH}/servicenow_incidents_jsonl}"

# Output directories (host paths). Default to job-scoped scratch folders.
KEPT_DIR="${KEPT_DIR:-${SCRATCH}/servicenow_incidents_processed/qa_parquet}"
REJECTED_DIR="${REJECTED_DIR:-${SCRATCH}/servicenow_incidents_processed/llm_rejected_parquet}"

# Container mount points for the pipeline.
MOUNT_TICKETS="/tickets"
MOUNT_KEPT="/kept"
MOUNT_REJECTED="/rejected"

mkdir -p "${KEPT_DIR}" "${REJECTED_DIR}"

export TICKETS_JSONL_DIR KEPT_DIR REJECTED_DIR MOUNT_TICKETS MOUNT_KEPT MOUNT_REJECTED

#------------------------------------------------------------------------------
# LLM pipeline runtime args (override via environment before sbatch)
#------------------------------------------------------------------------------
LLM_CONFIG="${LLM_CONFIG:-config/llm_pipeline.toml}"
LIMIT_TICKETS="${LIMIT_TICKETS:-}"

export LLM_CONFIG LIMIT_TICKETS

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

podman_llm() {
    local extra=()
    if [[ "${PODMAN_WITH_RAY_TMPDIR:-0}" == "1" ]]; then
      extra=(-v "${RAY_TEMP_DIR}:/tmp/ray")
    fi

    # Run the Ray LLM container
    podman-hpc run \
    --rm \
    -u 0 \
    --group-add keep-groups \
    --gpu \
    --nccl \
    --net host \
    --shm-size=40GB \
    -v "${HF_DIR}:/home/ray/.cache/huggingface" \
    -v "${PWD}:/workdir" \
    -v "${TICKETS_JSONL_DIR}:${MOUNT_TICKETS}" \
    -v "${KEPT_DIR}:${MOUNT_KEPT}" \
    -v "${REJECTED_DIR}:${MOUNT_REJECTED}" \
    "${extra[@]}" \
    -e "HF_TOKEN=${HF_KEY}" \
    -e "RAY_ADDRESS=${RAY_ADDRESS:-}" \
    -w /workdir \
    ${RAY_IMAGE} \
    "$@"
}
export -f podman_llm

# Get node list
log "Job ID: ${SLURM_JOB_ID}"
log "Allocated nodes: ${SLURM_JOB_NUM_NODES}"

# Get head and worker nodes
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
WORKER_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2)

log "Head node: ${HEAD_NODE}"
log "Worker nodes: ${WORKER_NODES}"

# Get head node IP address
export HEAD_NODE_IP=$(getent hosts "${HEAD_NODE}" | awk '{print $1}')
log "Head node IP: ${HEAD_NODE_IP}"

# Ray address for workers to connect
export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

#==============================================================================
# Start Ray Head Node
#==============================================================================
log "Starting Ray head node on ${HEAD_NODE}..."

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 -w "$HEAD_NODE" \
    bash -lc "
        PODMAN_WITH_RAY_TMPDIR=1 podman_llm \
            ray start \
            --head \
            --node-ip-address="${HEAD_NODE_IP}" \
            --port="${RAY_PORT}" \
            --dashboard-host=0.0.0.0 \
            --dashboard-port="${RAY_DASHBOARD_PORT}" \
            --block " &

HEAD_PID=$!
log "Ray head started with PID ${HEAD_PID}"

#==============================================================================
# Wait for Head Node to be Ready
#==============================================================================
log "Waiting for Ray head node to be ready (timeout: ${HEAD_NODE_TIMEOUT}s)..."

RAY_NODES_URL="http://${HEAD_NODE_IP}:${RAY_DASHBOARD_PORT}/api/v0/nodes"

SECONDS=0

while (( SECONDS < HEAD_NODE_TIMEOUT )); do
    if curl -sf --connect-timeout 2 --max-time 5 "$RAY_NODES_URL" | \
       jq -e '
         .data?.result?.result? // []
         | any(.is_head_node == true and .state == "ALIVE")
       ' >/dev/null 2>&1; then
        log "Ray head node is ready!"
        break
    fi

    sleep 1
done

if (( SECONDS >= HEAD_NODE_TIMEOUT )); then
    log "ERROR: Ray head node failed to start within ${HEAD_NODE_TIMEOUT} seconds"
    exit 1
fi

#==============================================================================
# Start Ray Worker Nodes
#==============================================================================
if [ -n "$WORKER_NODES" ]; then
    log "Starting Ray workers on: ${WORKER_NODES}"

    for worker in $WORKER_NODES; do
        log "Starting worker on ${worker}..."
        srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 -w "$worker" \
            bash -lc "
                podman_llm \
                    ray start \
                    --address="${RAY_ADDRESS}" \
                    --block " &
    done

    # Wait for all workers to connect (via Ray dashboard REST API)
    log "Waiting for all workers to connect (timeout: ${WORKER_TIMEOUT}s)..."
    EXPECTED_NODES=${SLURM_JOB_NUM_NODES}

    RAY_NODES_URL="http://${HEAD_NODE_IP}:${RAY_DASHBOARD_PORT}/api/v0/nodes"

    SECONDS=0
    CONNECTED_NODES=0

    while (( SECONDS < WORKER_TIMEOUT )); do
        # Count ALIVE nodes reported by the dashboard
        CONNECTED_NODES=$(
            curl -sf --connect-timeout 2 --max-time 5 "$RAY_NODES_URL" | \
            jq -r '
              .data?.result?.result? // []
              | map(select(.state == "ALIVE"))
              | length
            ' 2>/dev/null || echo "0"
        )

        if [ "$CONNECTED_NODES" -eq "$EXPECTED_NODES" ]; then
            log "All ${EXPECTED_NODES} nodes connected!"
            break
        fi

        sleep 1
    done

    if (( SECONDS >= WORKER_TIMEOUT )); then
        log "WARNING: Only ${CONNECTED_NODES}/${EXPECTED_NODES} nodes connected after ${WORKER_TIMEOUT} seconds"
    fi
else
    log "No worker nodes allocated (single-node cluster)"
fi


#==============================================================================
# Verify Cluster Status
#==============================================================================
log "Verifying cluster status..."
podman_llm ray status --address "${RAY_ADDRESS}"

log "Ray cluster is ready!"
log "Dashboard available at: http://${HEAD_NODE_IP}:${RAY_DASHBOARD_PORT}"
log "Ray address: ${RAY_ADDRESS}"

#==============================================================================
# Run ServiceNow ticket pipeline
#==============================================================================
log "Running pipeline..."
log "TICKETS_JSONL_DIR=${TICKETS_JSONL_DIR} (mounted at ${MOUNT_TICKETS})"
log "KEPT_DIR=${KEPT_DIR} (mounted at ${MOUNT_KEPT})"
log "REJECTED_DIR=${REJECTED_DIR} (mounted at ${MOUNT_REJECTED})"

# Edit PIPELINE_ARGS to add/remove flags as needed.
PIPELINE_ARGS=(
  --input "${MOUNT_TICKETS}"
  --output-dir "${MOUNT_KEPT}"
  --rejected-dir "${MOUNT_REJECTED}"
  --llm-config "${LLM_CONFIG}"
  --overwrite
  --ray-address "${RAY_ADDRESS}"
  --log-level INFO
)

if [[ -n "${LIMIT_TICKETS}" ]]; then
  PIPELINE_ARGS+=(--limit-tickets "${LIMIT_TICKETS}")
fi

PODMAN_WITH_RAY_TMPDIR=1 podman_llm python3.11 scripts/ray_servicenow_ticket_pipeline.py "${PIPELINE_ARGS[@]}"
