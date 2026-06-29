# Nivel 4 — Heatmap de movimiento + Streamlit Dashboard

Agrega acumulación de posiciones de personas para generar heatmaps de calor
en tiempo real y un dashboard interactivo con Streamlit.

## Qué hace

- Detecta y rastrea personas con YOLOv8 + DeepSORT.
- Acumula centroides de cada persona en un array 2D (HeatmapGenerator).
- Superpone el heatmap coloreado (colormap TURBO) sobre el video en vivo.
- Exporta métricas, CSV de dwell time y PNG del heatmap al cerrar.
- Dashboard Streamlit con feed en vivo, heatmap acumulado, reporte y descarga.

## Instalación

```bash
pip install -r requirements.txt
```

## Cómo correr

### Tracker standalone (webcam o video)

```bash
# Webcam
python heatmap_tracker.py

# Archivo de video
python heatmap_tracker.py --source assets/sample.mp4

# Guardar video procesado
python heatmap_tracker.py --save
```

### Dashboard Streamlit

```bash
streamlit run app.py
```

Abre `http://localhost:8501` en el navegador, elige la fuente de video y
presiona **▶ Start**.

## Atajos de teclado (heatmap_tracker.py)

| Tecla | Acción |
|-------|--------|
| `H`   | Mostrar / ocultar heatmap superpuesto |
| `S`   | Guardar snapshot PNG del heatmap actual en `output/` |
| `Q`   | Salir y guardar resultados finales |

## Outputs generados

```
output/
├── heatmap_final_YYYYMMDD_HHMMSS.png   ← heatmap acumulado final
├── heatmap_HHMMSS.png                  ← snapshots manuales con [S]
├── dwell_YYYYMMDD_HHMMSS.csv           ← tiempo de permanencia por persona
└── metrics_YYYYMMDD_HHMMSS.json        ← métricas agregadas de la sesión
```

### Estructura del JSON de métricas

```json
{
  "total_persons": 12,
  "avg_dwell_seconds": 47.3,
  "peak_zone": { "x": 320, "y": 240, "radius": 40 },
  "session_duration_seconds": 183.5
}
```

## Explicación del heatmap

El heatmap usa el colormap **TURBO** de OpenCV:

| Color | Significado |
|-------|-------------|
| 🔵 Azul / morado | Zona fría — pocas personas pasaron por aquí |
| 🟢 Verde / cian  | Actividad moderada |
| 🟡 Amarillo       | Zona caliente — tráfico frecuente |
| 🔴 Rojo / blanco  | Zona muy caliente — máxima densidad de personas |

Cuanto más tiempo y más personas pasan por un punto del frame,
más cálido (hacia el rojo) se vuelve ese píxel.

## Dashboard Streamlit — Pestañas

| Pestaña | Contenido |
|---------|-----------|
| 📷 Live Feed | Video procesado con heatmap superpuesto en tiempo real |
| 🔥 Heatmap   | Imagen estática del heatmap acumulado (actualiza cada 5 s) |
| 📊 Report    | Tabla de dwell times por persona + histograma Plotly |
| ⬇ Export    | Descarga CSV y JSON con los resultados de la sesión |

## Screenshot

```
┌──────────────────────────────────────────────────┐
│  🔥 Heatmap Analytics Dashboard                   │
│  Unique persons: 8   Avg dwell: 34.2s   Peak: (310,200) │
├──────────┬──────────┬──────────┬──────────────────┤
│ Live Feed│ Heatmap  │ Report   │ Export           │
│  [video] │ [imagen] │ [tabla]  │ [botones]        │
└──────────┴──────────┴──────────┴──────────────────┘
```

## Arquitectura

```
heatmap_generator.py   ← clase HeatmapGenerator (numpy accumulator)
heatmap_tracker.py     ← YOLO + DeepSORT + HeatmapGenerator (OpenCV window)
app.py                 ← Streamlit dashboard (usa YOLO + DeepSORT + HeatmapGenerator)
```
