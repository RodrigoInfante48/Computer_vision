# Nivel 1 — Detección básica de personas con YOLOv8

Detecta personas en tiempo real desde webcam o archivo de video usando YOLOv8n. Dibuja bounding boxes verdes y muestra FPS + conteo por frame en pantalla.

## Instalación y uso

```bash
pip install -r requirements.txt
python detect.py                          # webcam (por defecto)
python detect.py --source ruta/video.mp4  # archivo de video
python detect.py --source 0 --save        # webcam + guardar MP4
```

El video procesado se guarda en `output/detection_output.mp4` cuando se usa `--save`.

## Screenshot

<!-- Agregar captura de pantalla aquí -->
