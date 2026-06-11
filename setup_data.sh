#!/usr/bin/env bash
set -euo pipefail

RECORD_ID=15724986
FILE=data_bundle.tar.gz
HASH=f2aefaae33d855c5152416b04af77156318611916c1f26fa5354dec558e0ba9f

# Download data from Zenodo: https://zenodo.org/records/15724986
curl -L -o "$FILE" \
  "https://zenodo.org/record/${RECORD_ID}/files/${FILE}?download=1"

# verify download and extract
echo "$HASH  $FILE" | shasum -a 256 -c -     # verification
tar -xzf "$FILE"

# move period_results/ to figures/period_results/
cp -r data/period_results figures/
