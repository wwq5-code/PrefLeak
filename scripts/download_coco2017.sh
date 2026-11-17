#!/usr/bin/env bash
set -euo pipefail

COCO_DIR=${1:-${PREFLEAK_COCO_ROOT:-"$PWD/data/coco2017"}}
LOG_FILE="$COCO_DIR/download.log"

mkdir -p "$COCO_DIR"
exec >>"$LOG_FILE" 2>&1

echo "[$(date '+%F %T %z')] Starting COCO 2017 download"
echo "$$" >"$COCO_DIR/download.pid"
cd "$COCO_DIR"

wget --no-proxy --no-check-certificate -c --tries=0 --timeout=30 --progress=dot:giga \
  https://images.cocodataset.org/zips/train2017.zip

wget --no-proxy --no-check-certificate -c --tries=0 --timeout=30 --progress=dot:giga \
  https://images.cocodataset.org/zips/val2017.zip

wget --no-proxy --no-check-certificate -c --tries=0 --timeout=30 --progress=dot:giga \
  https://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "[$(date '+%F %T %z')] Downloads complete; extracting archives"
unzip -q -o train2017.zip
unzip -q -o val2017.zip
unzip -q -o annotations_trainval2017.zip

date >COCO2017_READY
echo "[$(date '+%F %T %z')] COCO 2017 download and extraction complete"
