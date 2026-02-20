#!/usr/bin/env bash
set -euo pipefail

# FilinGPT - EDGAR Bronze Loader (immutable raw source layer)

readonly USER_AGENT="FilinGPT research costakevinn.ml@gmail.com"
readonly BASE_URL="https://www.sec.gov/Archives/edgar/data"
readonly BRONZE_DIR="data/bronze"
readonly SLEEP_SEC="0.4"

mkdir -p "$BRONZE_DIR"

strip_dashes() {
  echo "${1//-/}"
}

download() {
  local cik="$1"
  local accession="$2"   # e.g. 0000320193-23-000106
  local output="$3"

  local accession_nodash
  accession_nodash="$(strip_dashes "$accession")"

  local url="${BASE_URL}/${cik}/${accession_nodash}/${accession}.txt"

  echo "Downloading -> ${output}"

  curl --fail --location \
       --retry 6 \
       --retry-delay 2 \
       --retry-all-errors \
       --compressed \
       --silent --show-error \
       -A "$USER_AGENT" \
       -H "Accept: text/plain,*/*" \
       "$url" \
       -o "$output"

  sleep "$SLEEP_SEC"
}

download "320193"  "0000320193-23-000106" "${BRONZE_DIR}/apple_2023_10k.txt"
download "1018724" "0001018724-23-000004" "${BRONZE_DIR}/amazon_2022_10k.txt"
download "789019"  "0000950170-23-035122" "${BRONZE_DIR}/microsoft_2023_10k.txt"

echo "Bronze layer complete."