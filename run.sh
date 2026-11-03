#!/usr/bin/env bash
# One command: the whole pipeline, after `make setup` (or with an environment
# that already satisfies requirements.txt on PATH as python3).
#   bash run.sh            full pipeline
#   bash run.sh --quick    reduced grid (smoke test, ~5 minutes)
#   bash run.sh --setup    create .venv and install the pinned requirements first
#
# This script deliberately does not guess at an interpreter: if the dependencies
# are missing it stops with the command that installs them, rather than failing
# halfway through with an ImportError.
set -euo pipefail
cd "$(dirname "$0")"
export TSFDI_ROOT="$PWD"
export PYTHONPATH="$PWD/src"

if [[ "${1:-}" == "--setup" ]]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
  export PATH="$PWD/.venv/bin:$PATH"
  shift || true
fi

if ! python3 -c "import numpy, pandas, scipy, sklearn, pywt, cryptography" 2>/dev/null; then
  echo "run.sh: the pinned dependencies are not installed for $(command -v python3)." >&2
  echo "        run 'make setup' (then 'source .venv/bin/activate'), or 'bash run.sh --setup'." >&2
  exit 1
fi

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then QUICK="--quick"; fi

python3 -m tsfdi data-verify
python3 -m tsfdi sites
python3 -m tsfdi gonogo --no-mc
python3 -m tsfdi calibrate
python3 -m tsfdi experiments $QUICK
python3 -m tsfdi report-figures
python3 -m tsfdi report-text
python3 -m tsfdi bench --n 20000
python3 -m tsfdi report-static
python3 -m tsfdi verify --tol 0.05 || echo "verify: results differ from expected_results.json (see above)"
echo "== done: see results/ (and 'make report-pdf' for docs/report.pdf) =="
