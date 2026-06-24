# Computer Vision System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv)

> Sistema progresivo de análisis de video con IA usando cámara. Cada nivel agrega una capa de inteligencia sobre el anterior.

---

## Niveles del sistema

| Nivel | Carpeta | Descripción |
|-------|---------|-------------|
| 1 | `01_basic_detection/` | Detección básica de objetos en tiempo real con webcam usando YOLOv8 |
| 2 | `02_people_counter/` | Contador de personas con zonas de entrada/salida y línea de conteo |
| 3 | `03_dwell_time/` | Tiempo de permanencia por persona usando tracking con IDs únicos (DeepSORT) |
| 4 | `04_heatmap_dashboard/` | Heatmap de movimiento + dashboard interactivo en Streamlit |
| 5 | `05_full_system/` | Sistema completo con base de datos, alertas y analytics en tiempo real |

---

## Instalación global

```bash
# 1. Clonar el repositorio
git clone https://github.com/RodrigoInfante48/Computer_vision.git
cd Computer_vision

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Getting Started

Empieza con el **Nivel 1** para verificar que tu entorno funciona correctamente:

```bash
cd 01_basic_detection
python detect.py
```

Consulta el `README.md` dentro de cada carpeta para instrucciones específicas de cada nivel.

---

## Estructura del proyecto

```
Computer_vision/
├── CLAUDE.md                ← Contexto para Claude Code
├── README.md                ← Este archivo
├── requirements.txt         ← Dependencias globales
├── assets/                  ← Videos de prueba y recursos compartidos
├── 01_basic_detection/      ← Nivel 1
├── 02_people_counter/       ← Nivel 2
├── 03_dwell_time/           ← Nivel 3
├── 04_heatmap_dashboard/    ← Nivel 4
└── 05_full_system/          ← Nivel 5
```

---

## Autor

**Rod (Rodrigo Infante)** — Senior BA / Analytics Engineer / DDI Founder
GitHub: [@RodrigoInfante48](https://github.com/RodrigoInfante48)

---

## License

MIT
