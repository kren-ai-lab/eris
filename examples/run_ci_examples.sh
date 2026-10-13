#!/usr/bin/env bash
set -euo pipefail

for example in examples/[0-9][0-9]_*.py; do
    MPLBACKEND=Agg uv run python "$example"
done
