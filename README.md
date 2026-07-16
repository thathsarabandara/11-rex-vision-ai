# 11 — REX Vision AI Service

The **visual-perception engine** of the REX Smart Home Robot Platform.

Receives live camera frames or uploaded images, runs configurable AI capabilities, maintains registered faces and gesture profiles, stores embeddings in Qdrant, publishes structured vision events to Kafka, and exposes real-time detection metadata via REST API and WebSocket.

---

## Architecture Overview

```
Camera → CameraWorker → frame_queue → InferenceWorker
                                           │
                          ┌────────────────┼──────────────────────┐
                          ▼                ▼                       ▼
                   ObjectPipeline    FacePipeline          GesturePipeline
                   MovementPipeline  ExpressionPipeline    ScenePipeline
                          │
                          ▼
                   VisionState (Redis)
                     │         │
              WebSocket    Kafka Topics
```

---

## AI Capabilities

| Capability | Model/Library | Status |
|---|---|---|
| Object Detection | YOLOv8 (Ultralytics) | ✅ Ready |
| Multi-Object Tracking | ByteTrack via Ultralytics | ✅ Ready |
| Face Detection | InsightFace (buffalo_sc) | ✅ Ready |
| Face Recognition | InsightFace 512-d embeddings + Qdrant | ✅ Ready |
| Expression Estimation | ONNX (FER-2013 fine-tuned) | ✅ Ready |
| Hand Landmark Extraction | MediaPipe Hands | ✅ Ready |
| Gesture Recognition | Cosine similarity vs Qdrant | ✅ Ready |
| Pose Estimation | MediaPipe Pose | ✅ Ready |
| Fall Detection | Rule-based pose transition | ✅ Ready |
| Movement Analysis | Bounding-box trajectory | ✅ Ready |
| Scene Understanding | Rule-based aggregation | ✅ Ready |
| Scene Description | Rule-based / optional model | ✅ Ready |
| Low-light Detection | Pixel mean threshold | ✅ Ready |
| Visual Memory Search | Qdrant semantic search | ✅ Ready |

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI 0.115 |
| Runtime | Python 3.11 + uvicorn (uvloop) |
| Database | MySQL 8 via SQLAlchemy 2 + asyncmy |
| Migrations | Alembic |
| Cache | Redis 7 (async) |
| Messaging | Kafka via aiokafka |
| Vector DB | Qdrant |
| Object Storage | MinIO |
| Auth | PyJWT (HS256) |
| Observability | Prometheus + structlog |

---

## Quick Start

### 1. Clone and configure
```bash
cp .env.example .env
# Edit .env with your settings
```

### 2. Start infrastructure
```bash
docker compose up -d mysql redis qdrant minio
```

### 3. Run migrations
```bash
alembic upgrade head
```

### 4. Start the service
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8011 --reload
```

### 5. OpenAPI docs
Navigate to `http://localhost:8011/docs`

---

## Key API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/robots/{id}/vision/latest` | Latest vision state |
| GET/PUT | `/api/v1/robots/{id}/vision/features` | Feature flag config |
| POST | `/api/v1/robots/{id}/vision/faces` | Register face |
| GET/PATCH/DELETE | `/api/v1/robots/{id}/vision/faces/{fid}` | Face profile CRUD |
| POST | `/api/v1/robots/{id}/vision/gestures` | Register gesture |
| POST/DELETE/GET | `/api/v1/robots/{id}/vision/tracking` | Person tracking target |
| GET/PUT | `/api/v1/robots/{id}/vision/models/{key}/state` | Enable/disable model |
| POST | `/api/v1/robots/{id}/vision/training/jobs` | Start training job |
| POST | `/api/v1/robots/{id}/vision/memory/search` | Visual memory search |
| WS | `/api/v1/ws/robots/{id}/vision` | Live vision stream |
| GET | `/internal/v1/robots/{id}/vision/context` | Voice/Agent context |
| GET | `/health/live`, `/health/ready`, `/health/gpu` | Health checks |

---

## Kafka Topics Published

| Topic | Description |
|---|---|
| `rex.vision.state.updated` | Full vision state every frame |
| `rex.vision.face.recognized` | Known face appeared |
| `rex.vision.unknown.person` | Unknown person appeared |
| `rex.vision.gesture.detected` | Stable gesture emitted |
| `rex.vision.fall.candidate` | Fall detection alert |
| `rex.vision.low.light` | Low-light warning |
| `rex.vision.context.updated` | Voice assistant context |
| `rex.vision.training.completed` | Training job completed |
| `rex.vision.training.failed` | Training job failed |

---

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests with coverage
pytest tests/ --asyncio-mode=auto --cov=app --cov-report=term-missing --cov-fail-under=90

# Lint
ruff check app/ tests/
```

---

## Training Scripts

```bash
# Object detection fine-tune
python training/object_detection/train.py --data data.yaml --model yolov8s.pt --epochs 50

# Object detection evaluation
python training/object_detection/evaluate.py --model runs/train/weights/best.pt --data data.yaml

# Gesture classifier training
python training/gestures/train.py --data gestures.csv --output models/gesture.pt

# Gesture classifier evaluation
python training/gestures/evaluate.py --model models/gesture.pt --data gestures.csv
```

---

## GPU Support

Set `DEVICE=cuda` (or `DEVICE=0`) in `.env`. The service detects CUDA availability at runtime and falls back to CPU gracefully if CUDA is unavailable.

For GPU Docker builds install `ultralytics[gpu]` and use the NVIDIA base image:
```bash
docker build -f Dockerfile.gpu -t rex-vision-ai:gpu .
```

---

## Integrations

| Service | Purpose |
|---|---|
| Robot Service | Ownership verification |
| Auth Service | JWT validation |
| Voice Assistant | `/internal/v1/.../vision/context` |
| Agent Runtime | `/internal/v1/.../vision/context` |
| Event Engine | Vision events via Kafka |
| Sensor Fusion | Low-light, fall events |
| API Gateway | Public routes proxied |

---

## Directory Structure

```
11-rex-vision-ai/
├── app/
│   ├── ai/                 # AI model wrappers (lazy-loading)
│   ├── config/             # Settings, DB, Redis, Kafka, Qdrant, MinIO
│   ├── middleware/         # Auth, rate limit, request ID, error handler
│   ├── models/             # SQLAlchemy ORM models
│   ├── pipelines/          # Per-frame AI pipeline steps
│   ├── routes/             # FastAPI routers
│   ├── schemas/            # Pydantic request/response schemas
│   ├── services/           # Business logic services
│   ├── utils/              # Image, embedding, validation, date helpers
│   └── workers/            # Camera, inference, training, cleanup workers
├── migrations/             # Alembic migration scripts
├── tests/                  # pytest test suite
├── training/               # Standalone training scripts
│   ├── common/
│   ├── gestures/
│   └── object_detection/
├── .github/workflows/      # GitHub Actions CI/CD
├── Dockerfile
├── Jenkinsfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── alembic.ini
```
