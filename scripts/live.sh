#!/bin/sh
# Живой режим + дашборд. Frigate поднимается отдельно: docker compose up -d
cd "$(dirname "$0")/.." || exit 1
exec .venv/bin/python -m storevision.live "${1:-configs/live_demo.yaml}" --port "${PORT:-8088}"
