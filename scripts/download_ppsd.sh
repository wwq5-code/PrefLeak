#!/usr/bin/env bash
set -euo pipefail

PPSD_DIR=${1:-${PREFLEAK_PPSD_ROOT:-"$PWD/data/PPSD"}}
LOG_FILE="$PPSD_DIR/download.log"

mkdir -p "$PPSD_DIR"
exec >>"$LOG_FILE" 2>&1

echo "[$(date '+%F %T %z')] Starting PPSD download"
echo "$$" >"$PPSD_DIR/download.pid"

download_drive_file() {
    local file_id=$1
    local filename=$2
    local expected_bytes=$3
    local output_path="$PPSD_DIR/$filename"
    local download_url="https://drive.usercontent.google.com/download?id=$file_id&export=download&confirm=t"

    until curl \
        --location \
        --fail \
        --continue-at - \
        --connect-timeout 30 \
        --speed-time 120 \
        --speed-limit 1024 \
        --output "$output_path" \
        "$download_url"; do
        echo "[$(date '+%F %T %z')] Download interrupted for $filename; retrying in 15 seconds"
        sleep 15
    done

    local actual_bytes
    actual_bytes=$(stat -c '%s' "$output_path")
    if [[ "$actual_bytes" != "$expected_bytes" ]]; then
        echo "Size check failed for $filename: expected $expected_bytes, got $actual_bytes" >&2
        return 1
    fi
    unzip -tq "$output_path"
    echo "[$(date '+%F %T %z')] Verified $filename ($actual_bytes bytes)"
}

download_drive_file 1ZlrPUh6Rn_fwIA0t8JdLAcmg1n8ihMT6 img_src.zip 1397006401
download_drive_file 1R21XEfrX7TfEcvgu2loc5hXNo4HD2h6T responses.zip 2728259

echo "[$(date '+%F %T %z')] Downloads complete; extracting archives"
mkdir -p "$PPSD_DIR/img_src" "$PPSD_DIR/responses"
unzip -q -o "$PPSD_DIR/img_src.zip" -d "$PPSD_DIR/img_src"
unzip -q -o "$PPSD_DIR/responses.zip" -d "$PPSD_DIR/responses"

date >"$PPSD_DIR/PPSD_READY"
echo "[$(date '+%F %T %z')] PPSD download and extraction complete"
