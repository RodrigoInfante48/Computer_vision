# Nivel 3 — Dwell Time Tracker

Mide cuánto tiempo permanece cada persona detectada en cámara o en zonas específicas,
usando DeepSORT para tracking robusto con IDs persistentes.

## ¿Por qué DeepSORT y no un tracker básico?

| Característica | Tracker básico (IoU) | DeepSORT |
|---|---|---|
| Asignación de ID | Por solapamiento de bboxes | Apariencia visual + movimiento (Kalman) |
| Re-identificación | No — pierde el ID al ocluirse | Sí — recupera el mismo ID tras oclusión breve |
| Consistencia de IDs | Baja en multitudes | Alta; cada persona mantiene su ID |
| Tiempo de permanencia | Subestimado por pérdidas | Preciso aunque la persona salga y vuelva |

DeepSORT combina un filtro de Kalman (predicción de posición) con un descriptor de
apariencia CNN, lo que permite re-identificar personas incluso cuando se cruzan o
quedan temporalmente fuera de cuadro.

## Colores de los bounding boxes

| Color | Significado | Tiempo en cámara |
|---|---|---|
| Verde | Visita corta | < 30 segundos |
| Amarillo | Visita media | 30 – 120 segundos |
| Rojo | Visita larga | > 120 segundos |

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Webcam por defecto
python tracker.py

# Archivo de video
python tracker.py --source assets/sample.mp4

# Guardar video procesado en output/
python tracker.py --source assets/sample.mp4 --save

# Con zonas nombradas (JSON)
python tracker.py --source assets/sample.mp4 --zones zones.json
```

### Formato del archivo de zonas (zones.json)

```json
{
  "entrada":  [[0,   0],   [320, 480]],
  "mostrador": [[320, 100], [640, 480]]
}
```

Cada zona se define como dos puntos: esquina superior-izquierda y esquina
inferior-derecha del rectángulo. Las coordenadas son píxeles relativos al frame.

## Reporte CSV

Al cerrar con `q` se genera automáticamente `output/dwell_report_YYYYMMDD_HHMMSS.csv`
con las siguientes columnas:

| Columna | Descripción |
|---|---|
| `track_id` | ID único asignado por DeepSORT |
| `first_seen_frame` | Frame en que apareció por primera vez |
| `last_seen_frame` | Último frame en que fue detectada |
| `total_frames` | Frames totales donde fue detectada activamente |
| `dwell_seconds` | Tiempo total de permanencia en segundos |
| `avg_zone` | Zona donde más tiempo estuvo (si se usaron zonas) |

## Casos de uso reales

- **Cafeterías / restaurantes**: detectar mesas con clientes esperando mucho tiempo,
  optimizar turnos de atención.
- **Retail / tiendas**: identificar zonas de alta permanencia para optimizar layout,
  medir efectividad de exhibiciones.
- **Salas de espera**: monitorear tiempo promedio de espera, generar alertas si supera
  un umbral definido.
- **Seguridad**: detectar personas estacionadas demasiado tiempo en áreas sensibles.

## Panel lateral en tiempo real

Durante la ejecución se muestra un panel en el lado derecho del frame con:
- Top 5 personas por mayor tiempo en cámara
- Total de personas únicas detectadas
- Tiempo promedio de permanencia

## Estructura de archivos

```
03_dwell_time/
├── tracker.py          ← Script principal, loop de detección y tracking
├── dwell_analyzer.py   ← Clase DwellAnalyzer, lógica de medición de tiempo
├── report_generator.py ← Generación de reporte CSV y resumen en consola
├── requirements.txt
├── README.md
└── output/             ← Generado automáticamente (ignorado por git)
    ├── dwell_report_*.csv
    └── dwell_*.mp4     (si se usa --save)
```
