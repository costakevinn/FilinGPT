#!/usr/bin/env bash
set -euo pipefail

# FilinGPT - PREP runner (Docker)

IMAGE="filingpt"
DOCKER="docker run --rm -v \"$(pwd):/app\" -w /app ${IMAGE} python -m"

SKIP_TESTS=0
CLEAN=0

for arg in "$@"; do
  case "$arg" in
    --skip-tests) SKIP_TESTS=1 ;;
    --clean) CLEAN=1 ;;
    *)
      echo "[ERR] Unknown argument: $arg"
      echo "Usage: scripts/run_prep.sh [--clean] [--skip-tests]"
      exit 1
      ;;
  esac
done

if [[ "$CLEAN" -eq 1 ]]; then
  echo "[INFO] Cleaning training outputs..."
  rm -rf data/training
fi

echo "[INFO] PREP: build dataset"
eval ${DOCKER} prep.00_build_dataset

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "[INFO] PREP: test dataset"
  eval ${DOCKER} prep.01_test_dataset
fi

echo "[INFO] PREP: chunk dataset"
eval ${DOCKER} prep.02_chunk_dataset

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "[INFO] PREP: test chunks"
  eval ${DOCKER} prep.03_test_chunks
fi

echo "[INFO] PREP: tokenize chunks"
eval ${DOCKER} prep.04_tokenize_chunks

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "[INFO] PREP: test tokens"
  eval ${DOCKER} prep.05_test_tokens
fi

echo "[INFO] PREP: build batches"
eval ${DOCKER} prep.06_build_batches

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "[INFO] PREP: test batches"
  eval ${DOCKER} prep.07_test_batches
fi

echo "[OK] PREP complete."