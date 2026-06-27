@echo off
REM ─────────────────────────────────────────────────────────
REM  Launcher — Level 2: People Counter  (Windows)
REM  Uso:
REM    run.bat                    → webcam (índice 0)
REM    run.bat --save             → webcam + guardar video
REM    run.bat --source video.mp4
REM    run.bat --source video.mp4 --save
REM ─────────────────────────────────────────────────────────

cd /d "%~dp0"

REM ── 1. Entorno virtual ───────────────────────────────────
IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo >>> Creando entorno virtual...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

REM ── 2. Dependencias ─────────────────────────────────────
echo >>> Verificando dependencias...
pip install -q -r requirements.txt

REM ── 3. Directorio de salida ──────────────────────────────
IF NOT EXIST "output" mkdir output

REM ── 4. Ejecutar ──────────────────────────────────────────
echo >>> Iniciando counter.py %*
echo     Presiona Q para salir.
echo.
python counter.py %*
