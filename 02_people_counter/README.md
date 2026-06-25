# Nivel 2 — People Counter with Virtual Zones

Detecta personas con YOLOv8 y cuenta cuántas cruzan líneas virtuales de entrada (IN) y salida (OUT), exportando cada evento a un CSV.

## Cómo funciona

### Detección de cruce de línea
Cada frame se trackea con `model.track()` (tracker interno de ultralytics), que asigna un ID numérico persistente a cada persona. El script guarda el centroide `(cx, cy)` de cada ID en el frame anterior. Cuando el `cy` actual cruza la coordenada Y de una zona:

```
prev_cy < line_y <= curr_cy   →  movimiento hacia abajo
curr_cy < line_y <= prev_cy   →  movimiento hacia arriba
```

Si el ID todavía no había sido contado para esa zona, se registra el evento y se agrega el ID a un `set` para evitar doble conteo.

- **Zona IN** — línea horizontal al 40 % del alto del frame (verde)
- **Zona OUT** — línea horizontal al 60 % del alto del frame (rojo)

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Webcam por defecto
python counter.py

# Archivo de video
python counter.py --source assets/video.mp4

# Webcam + guardar video procesado
python counter.py --save

# Zonas personalizadas (JSON)
python counter.py --zone-config my_zones.json
```

### Argumentos CLI

| Argumento | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--source` | str/int | `0` | Índice de cámara o ruta a video |
| `--save` | flag | off | Guarda video anotado en `output/` |
| `--zone-config` | path | None | JSON con zonas personalizadas |

## Customizar zonas

Crea un archivo JSON con coordenadas proporcionales (0.0 – 1.0):

```json
{
  "IN":  [[0.0, 0.35], [1.0, 0.35]],
  "OUT": [[0.0, 0.65], [1.0, 0.65]]
}
```

Pasa el archivo con `--zone-config mi_zona.json`. Las coordenadas se escalan automáticamente a la resolución real del video.

## Formato del CSV de salida

Archivo: `output/counts_YYYYMMDD_HHMMSS.csv`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `timestamp` | ISO 8601 | Fecha y hora exacta del evento |
| `event` | `IN` / `OUT` | Tipo de cruce detectado |
| `person_id` | int | ID único asignado por el tracker |
| `frame_number` | int | Frame en que ocurrió el cruce |

Ejemplo:

```
timestamp,event,person_id,frame_number
2024-03-15T10:23:01.452,IN,3,87
2024-03-15T10:23:04.118,OUT,3,179
2024-03-15T10:23:07.331,IN,5,261
```

## Salidas

```
02_people_counter/output/
├── counts_20240315_102301.csv   ← log de eventos
└── counts_20240315_102301.mp4   ← video anotado (si --save)
```
