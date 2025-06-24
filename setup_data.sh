#!/usr/bin/env bash
set -euo pipefail

RECORD_ID=15724986
FILE=data_bundle.tar.gz
HASH=603bab8f0c4658d032358e2011946394e7bb5fcada6e77ce4e13b9ffaa396a52

# Download data from Zenodo: https://zenodo.org/records/15724986
curl -L -o "$FILE" \
  "https://zenodo.org/record/${RECORD_ID}/files/${FILE}?download=1"

# verify download and extract
echo "$HASH  $FILE" | shasum -a 256 -c -     # verification
tar -xzf "$FILE"

# move period_results/ to figures/period_results/
mv data/period_results figures/
