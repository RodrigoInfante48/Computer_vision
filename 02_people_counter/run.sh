#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Launcher — Level 2: People Counter
#  Usage:
#    ./run.sh            → webcam (índice 0)
#    ./run.sh --save     → webcam + guardar video
#    ./run.sh --source video.mp4
#    ./run.sh --source video.mp4 --save
# ─────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Entorno virtual ───────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo ">>> Creando entorno virtual..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── 2. Dependencias ──────────────────────────────────────────
echo ">>> Verificando dependencias..."
pip install -q -r requirements.txt

# ── 3. Directorio de salida ──────────────────────────────────
mkdir -p output

# ── 4. Ejecutar ──────────────────────────────────────────────
echo ">>> Iniciando counter.py $*"
echo "    Presiona Q para salir."
echo ""
python counter.py "$@"
