#!/usr/bin/env bash
set -euo pipefail
"${PYTHON:-python}" scripts/smoke_test.py --device cpu --limit 20 "$@"
