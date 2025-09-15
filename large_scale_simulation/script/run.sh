#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../SimAI" && pwd)"

BINARY="${BINARY:-$PROJECT_ROOT/bin/SimAI_simulator}"
CONFIG="${CONFIG:-$SCRIPT_DIR/SimAI.conf}"
WORKLOAD_DIR="${WORKLOAD_DIR:-$SCRIPT_DIR/../workload}"
TOPO_DIR="${TOPO_DIR:-$SCRIPT_DIR/../topo}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../results}"
THREADS="${THREADS:-16}"

MODEL_SIZE=$1
TOPOLOGY=$2
GBS=$3
shift 3
BANDWIDTHS=("$@")

if [ -z "$MODEL_SIZE" ] || [ -z "$TOPOLOGY" ] || [ -z "$GBS" ] || [ ${#BANDWIDTHS[@]} -eq 0 ]; then
  echo "ERROR: Missing arguments."
  echo "Usage: $0 <MODEL_SIZE> <TOPOLOGY> <GBS> <BANDWIDTH1> [BANDWIDTH2] ..."
  echo "  MODEL_SIZE: 70B, 175B, or 405B"
  echo "  TOPOLOGY: dcn_512"
  echo "  GBS: global batch size"
  echo "  BANDWIDTH: one or more Gbps values"
  exit 1
fi

if [ ! -f "$BINARY" ]; then
  echo "ERROR: Binary not found at $BINARY"
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: Config file not found at $CONFIG"
  exit 1
fi

case "$MODEL_SIZE" in
  "7B")
    WORKLOAD="../workload/None-gpt_7B-world_size512-tp4-pp1-ep1-gbs${GBS}-mbs1-seq2048-MOE-False-GEMM-False-flash_attn-True.txt"
    OUTPUT_PREFIX="gpt_7b"
    ;;
  *)
    exit 1
    ;;
esac

case "$TOPOLOGY" in
  "dcn_512")
    TOPO_BASE="DCN+SingleToR_512g_1gps"
    OUTPUT_SUFFIX="dcn_512"
    ;;
  *)
    echo "ERROR: Unknown TOPOLOGY '$TOPOLOGY' (use 'dcn_512')."
    exit 1
    ;;
esac

if [ ! -f "$WORKLOAD" ]; then
  echo "ERROR: Workload file not found at $WORKLOAD"
  exit 1
fi

mkdir -p "$RESULTS_DIR"

for BW in "${BANDWIDTHS[@]}"; do
  echo "=================================================="
  echo "Running $MODEL_SIZE with GBS=$GBS and $BW Gbps on $TOPOLOGY..."
  echo "=================================================="

  TOPO_FILE="$TOPO_DIR/${TOPO_BASE}_${BW}Gbps_H100"
  if [ ! -f "$TOPO_FILE" ]; then
    echo "ERROR: Topology file not found at $TOPO_FILE"
    exit 1
  fi

  OUTPUT_NAME="${OUTPUT_PREFIX}_${OUTPUT_SUFFIX}"
  LOG_FILE="${OUTPUT_NAME}.log"

  sudo "$BINARY" \
    -t "$THREADS" \
    -w "$WORKLOAD" \
    -n "$TOPO_FILE" \
    -c "$CONFIG" \
    | sudo tee "$LOG_FILE"

  RUN_RESULTS_DIR="$RESULTS_DIR/${OUTPUT_NAME}_${GBS}gbs_${BW}gbps"
  mkdir -p "$RUN_RESULTS_DIR"

  cp "$LOG_FILE" \
     ncclFlowModel_EndToEnd.csv \
     ncclFlowModel_test1_dimension_utilization_0.csv \
     "$RUN_RESULTS_DIR/"

  echo "Results for $BW Gbps run copied to $RUN_RESULTS_DIR"
  echo
done

echo "All runs completed!"