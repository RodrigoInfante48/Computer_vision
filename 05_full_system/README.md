# Nivel 5 — Sistema Completo con SQLite, FastAPI y Alertas

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                       main.py                           │
│  Webcam/Video → YOLO → DeepSORT → AlertEngine           │
│                           │                             │
│                    SQLite (sessions.db)                 │
│                  sessions / detections / alerts         │
└────────────────────────┬────────────────────────────────┘
                         │ shared in-process state
┌────────────────────────▼────────────────────────────────┐
│                       api.py (FastAPI)                  │
│  GET /health                                            │
│  GET /sessions                                          │
│  GET /sessions/{id}/stats                               │
│  GET /sessions/{id}/detections                          │
│  GET /sessions/{id}/heatmap   ← returns PNG             │
│  GET /live/stats              ← métricas en tiempo real │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP POST (webhook)
            ┌────────────▼────────────┐
            │  Make.com / n8n / Slack │
            └─────────────────────────┘
```

## Configuración rápida

```bash
cd 05_full_system
cp .env.example .env
# Edita .env con tus valores
pip install -r requirements.txt
```

### Variables de entorno (`.env`)

| Variable | Default | Descripción |
|---|---|---|
| `CAMERA_INDEX` | `0` | Índice de la webcam (o ruta a archivo de video) |
| `MODEL_SIZE` | `yolov8n.pt` | Modelo YOLO (n/s/m/l/x) |
| `CONFIDENCE_THRESHOLD` | `0.4` | Confianza mínima para detecciones |
| `MAX_OCCUPANCY` | `10` | Límite de personas simultáneas antes de alertar |
| `DWELL_ALERT_SECONDS` | `300` | Segundos antes de disparar alerta de permanencia |
| `ALERT_WEBHOOK_URL` | *(vacío)* | URL del webhook para alertas |
| `DB_PATH` | `./output/sessions.db` | Ruta al archivo SQLite |

## Cómo correr

Abre **dos terminales** en la carpeta `05_full_system/`:

**Terminal 1 — Visión (cámara + tracking):**
```bash
python main.py
# Con video de prueba:
python main.py ../assets/sample.mp4
```

**Terminal 2 — API REST:**
```bash
uvicorn api:app --reload --port 8000
```

La API queda disponible en `http://localhost:8000`.
Documentación interactiva: `http://localhost:8000/docs`

## Endpoints — ejemplos curl

```bash
# Estado del sistema
curl http://localhost:8000/health

# Listar todas las sesiones
curl http://localhost:8000/sessions

# Stats de una sesión específica
curl http://localhost:8000/sessions/1/stats

# Todas las detecciones de una sesión
curl http://localhost:8000/sessions/1/detections

# Heatmap como PNG (guardarlo en disco)
curl http://localhost:8000/sessions/1/heatmap --output heatmap.png

# Métricas en vivo (mientras main.py corre)
curl http://localhost:8000/live/stats
```

## Alertas con Make.com

1. En Make.com crea un escenario con trigger **Webhooks → Custom Webhook**.
2. Copia la URL generada (ej: `https://hook.make.com/abc123`).
3. Pégala en `.env` como `ALERT_WEBHOOK_URL`.
4. El payload que recibirás tiene esta estructura:

```json
{
  "alert_type": "occupancy_exceeded",
  "timestamp": "2026-07-01T12:00:00+00:00",
  "value": 12,
  "threshold": 10,
  "session_id": 1
}
```

Tipos de alerta posibles: `occupancy_exceeded`, `dwell_exceeded`.

Para `dwell_exceeded` el payload incluye además `"track_id": <int>`.

5. Conecta módulos de Make para enviar emails, Slack, SMS, etc.

## Outputs generados

```
05_full_system/output/
├── sessions.db          ← Base de datos SQLite
└── session_1.avi        ← Video procesado con anotaciones
```
