# CLAUDE.md — Contexto del proyecto Computer Vision

## ¿Qué es este proyecto?
Sistema progresivo de análisis de video con IA usando cámara. Construido en Python.
Cada carpeta es un nivel independiente que demuestra una capacidad mayor.

## Stack tecnológico
- Python 3.10+
- YOLOv8 (ultralytics) — detección de objetos
- DeepSORT (deep_sort_realtime) — tracking multi-objeto con IDs únicos
- OpenCV (opencv-python) — captura de video y procesamiento de frames
- pandas — logging y analytics
- Streamlit — dashboards (Nivel 4+)

## Estructura del repo
```
Computer_vision/
├── CLAUDE.md                ← Este archivo. Léelo SIEMPRE antes de codear.
├── README.md                ← Documentación pública del repo
├── requirements.txt         ← Dependencias globales
├── assets/                  ← Videos de prueba y recursos compartidos
├── 01_basic_detection/      ← Nivel 1: Detección básica con webcam
├── 02_people_counter/       ← Nivel 2: Contador de personas con zonas
├── 03_dwell_time/           ← Nivel 3: Tiempo de permanencia por persona
├── 04_heatmap_dashboard/    ← Nivel 4: Heatmap + Streamlit dashboard
└── 05_full_system/          ← Nivel 5: Sistema completo con DB y alertas
```

## Convenciones de código
- Todos los scripts tienen un bloque `if __name__ == "__main__":`
- Variables de configuración siempre al tope del archivo (mayúsculas)
- Cada nivel tiene su propio README.md explicando qué hace y cómo correrlo
- Los outputs (videos procesados, CSVs) van en una carpeta `output/` dentro de cada nivel
- `.gitignore` excluye: `output/`, `*.pt` (modelos YOLO), `__pycache__/`, `.env`

## Reglas para Claude Code
- SIEMPRE leer este archivo al inicio de cada sesión
- NUNCA hardcodear rutas absolutas — usar `pathlib.Path` relativas
- SIEMPRE incluir un `requirements.txt` local por nivel además del global
- El modelo YOLO a usar por defecto es `yolov8n.pt` (más rápido)
- Preferir webcam (índice 0) como input por defecto, con fallback a archivo de video

## Autor
Rod (Rodrigo Infante) — Senior BA / Analytics Engineer / DDI Founder
GitHub: RodrigoInfante48
