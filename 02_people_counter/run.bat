@echo off
REM Launcher - Level 2: People Counter (Windows)
REM Uso:
REM   run.bat
REM   run.bat --save
REM   run.bat --source ruta\video.mp4
REM   run.bat --source ruta\video.mp4 --save

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo Creando entorno virtual...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Verificando dependencias...
pip install -q -r requirements.txt

IF NOT EXIST "output" mkdir output

echo Iniciando counter.py %*
echo Presiona Q para salir.
echo.
python counter.py %*
