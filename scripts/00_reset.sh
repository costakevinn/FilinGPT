#!/usr/bin/env bash
set -euo pipefail

# FilinGPT - Reset script
# Removes generated layers (silver/gold/training/artifacts)
# Optionally removes bronze layer too.

KEEP_BRONZE=1
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --all) KEEP_BRONZE=0 ;;
    --force) FORCE=1 ;;
    *)
      echo "[ERR] Unknown argument: $arg"
      echo "Usage: scripts/00_reset.sh [--all] [--force]"
      echo "  --all    also remove bronze layer"
      echo "  --force  skip confirmation"
      exit 1
      ;;
  esac
done

echo "----------------------------------------"
echo "FilinGPT reset"
echo "----------------------------------------"

echo "[INFO] Will remove:"
echo "  data/silver"
echo "  data/gold"
echo "  data/training"
echo "  data/artifacts"
echo "  __pycache__ directories"

if [[ "$KEEP_BRONZE" -eq 0 ]]; then
  echo "  data/bronze"
fi

if [[ "$FORCE" -eq 0 ]]; then
  echo ""
  read -rp "Continue? (y/N): " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

rm -rf data/silver data/gold data/training data/artifacts

if [[ "$KEEP_BRONZE" -eq 0 ]]; then
  rm -rf data/bronze
fi

# Remove Python caches
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "[OK] Reset complete."