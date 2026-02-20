#!/usr/bin/env bash
set -euo pipefail

# FilinGPT - ETL runner (Docker)

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
      echo "Usage: scripts/run_etl.sh [--clean] [--skip-tests]"
      exit 1
      ;;
  esac
done

if [[ "$CLEAN" -eq 1 ]]; then
  echo "[INFO] Cleaning silver/gold outputs..."
  rm -rf data/silver data/gold
fi

echo "[INFO] ETL: bronze validation"
eval ${DOCKER} etl.00_test_bronze

echo "[INFO] ETL: extract 10-K"
eval ${DOCKER} etl.01_extract_10k

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "[INFO] ETL: silver validation"
  eval ${DOCKER} etl.02_test_10k
fi

echo "[INFO] ETL: extract MD&A"
eval ${DOCKER} etl.03_extract_mda

if [[ "$SKIP_TESTS" -eq 0 ]]; then
  echo "[INFO] ETL: gold validation"
  eval ${DOCKER} etl.04_test_mda
fi

echo "[OK] ETL complete."