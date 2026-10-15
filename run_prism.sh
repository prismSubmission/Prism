#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash run_prism.sh -d <dataset> [-b 0|1] [-et 0|1|2] [-mos <mode>] [-op <output.csv>]

Datasets:
  parler_multi
  parler_single
  amazon_multi
  amazon_single

Defaults:
  -b 0
  -et 0
  -mos inference_first
  -op single_run_output/<dataset>/output.csv

Query selection:
  This script runs one query per invocation: the first valid line in
  dataset/<dataset>/<dataset>_ans.txt. To run a different query, move the
  target query line to the top of that workload file before launching the
  script.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
BUILD_DIR="$REPO_ROOT/build"
PRISM_BIN="$BUILD_DIR/Prism"

B=0
ET=0
MOS="inference_first"
DATASET=""
OUT_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b)
      B="${2:-}"
      shift 2
      ;;
    -et)
      ET="${2:-}"
      shift 2
      ;;
    -mos)
      MOS="${2:-}"
      shift 2
      ;;
    -d)
      DATASET="${2:-}"
      shift 2
      ;;
    -op)
      OUT_ARG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$DATASET" in
  parler_multi|parler_single|amazon_multi|amazon_single)
    ;;
  "")
    echo "Missing required argument: -d" >&2
    usage >&2
    exit 2
    ;;
  *)
    echo "Unsupported dataset: $DATASET" >&2
    usage >&2
    exit 2
    ;;
esac

case "$B" in
  0|1) ;;
  *)
    echo "Invalid -b value: $B" >&2
    exit 2
    ;;
esac

case "$ET" in
  0|1|2) ;;
  *)
    echo "Invalid -et value: $ET" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PRISM_BIN" ]]; then
  echo "Cannot find executable: build/Prism" >&2
  echo "Build the project first, for example: cmake -S . -B build && cmake --build build" >&2
  exit 1
fi

DATASET_ROOT="$REPO_ROOT/dataset/$DATASET"
ANS_FILE="$DATASET_ROOT/${DATASET}_ans.txt"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Missing dataset directory: dataset/$DATASET" >&2
  exit 1
fi

if [[ ! -f "$ANS_FILE" ]]; then
  echo "Missing workload file: dataset/$DATASET/${DATASET}_ans.txt" >&2
  exit 1
fi

# Single-query mode: Prism reads the first valid line from this workload file.
# To run another query, put that query line first in dataset/<dataset>/<dataset>_ans.txt.

OUT_REL="${OUT_ARG:-single_run_output/$DATASET/output.csv}"
OUT_REL="${OUT_REL#./}"

if [[ "$OUT_REL" = /* ]]; then
  OUT_CSV="$OUT_REL"
  BIN_OUT="$OUT_REL"
else
  OUT_CSV="$REPO_ROOT/$OUT_REL"
  BIN_OUT="../$OUT_REL"
fi

OUT_DIR="$(dirname "$OUT_CSV")"
mkdir -p "$OUT_DIR"
OUT_LOG="$OUT_DIR/output.log"
ERR_LOG="$OUT_DIR/output.err"

DATASET_DIR="../dataset/$DATASET"
ANS_ARG="$DATASET_DIR/${DATASET}_ans.txt"
GRAPH_FILE="$DATASET_ROOT/${DATASET}.graph"

if [[ ! -f "$GRAPH_FILE" ]]; then
  echo "Missing graph file: dataset/$DATASET/${DATASET}.graph" >&2
  exit 1
fi

COMMON_ARGS=(
  -d "$DATASET"
  -q "$ANS_ARG"
  --STRUCTURE X
  -et "$ET"
  -b "$B"
  -mos "$MOS"
  -op "$BIN_OUT"
)

if [[ "$DATASET" == parler_* ]]; then
  INTERNAL_ARGS=(
    -vr "$DATASET_DIR/id_mapping.csv"
    -rk "$DATASET_DIR/csv_data/comment.csv"
    -pM "$DATASET_DIR/csv_data/post.csv"
    -ml "$DATASET_DIR/core_nodes_config.json"
    -vp "$DATASET_DIR/id_mapping.csv"
  )
else
  FIRST_QUERY="$(awk 'NF >= 3 {print $1; exit}' "$ANS_FILE")"
  if [[ -z "$FIRST_QUERY" ]]; then
    echo "Cannot read first query from dataset/$DATASET/${DATASET}_ans.txt" >&2
    exit 1
  fi

  QUERY_STEM="${FIRST_QUERY%.graph}"
  VERTEX_SIZE="${QUERY_STEM#query_}"
  VERTEX_SIZE="${VERTEX_SIZE%%_*}"

  if [[ "$DATASET" == "amazon_single" ]]; then
    PREDICATE_COUNT=1
  else
    PREDICATE_COUNT=2
  fi

  MODEL_LOCATION="$DATASET_DIR/queryWithModel/ModelLocation/$PREDICATE_COUNT/$VERTEX_SIZE/$QUERY_STEM.csv"

  INTERNAL_ARGS=(
    -vr "$DATASET_DIR/vertex_review.csv"
    -rk "$DATASET_DIR/reviews_kcore.csv"
    -pM "$DATASET_DIR/meta_kcore.csv"
    -ml "$MODEL_LOCATION"
    -vp "$DATASET_DIR/vertex_product.csv"
    -ip "$DATASET_DIR/pic/images_kcore"
  )
fi

echo "[RUN] dataset=$DATASET b=$B et=$ET mos=$MOS"
echo "[OUT] $OUT_REL"

cd "$BUILD_DIR"

# Prefer the system C++ runtime when Conda provides an older libstdc++.
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_LIBRARY_PATH

"$PRISM_BIN" "${COMMON_ARGS[@]}" "${INTERNAL_ARGS[@]}" > "$OUT_LOG" 2> "$ERR_LOG"

echo "[OK] finished"
echo "[LOG] ${OUT_LOG#$REPO_ROOT/}"
