#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/research-os}
STAMP=$(date -u +%Y%m%d_%H%M%S)
OUT=${OUT:-$ROOT/backups}
mkdir -p "$OUT"

cd "$ROOT"

ARCHIVE="$OUT/research_os_control_${STAMP}.tgz"
LATEST="$OUT/research_os_control_latest.tgz"

tar -czf "$ARCHIVE" \
  queues docs schemas templates scripts workers \
  research/*/project.yaml research/*/deep_research research/*/paper research/*/blog \
  2>/dev/null || true

ln -sf "$(basename "$ARCHIVE")" "$LATEST"

echo "wrote $ARCHIVE"
echo "latest $LATEST"

if [[ -n "${S3_URI:-}" ]]; then
  aws s3 cp "$ARCHIVE" "$S3_URI/"
  aws s3 cp "$ARCHIVE" "$S3_URI/research_os_control_latest.tgz"
  echo "uploaded to $S3_URI"
fi
