# DrishtiSense

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-Async-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLO-Detection-111F68?style=for-the-badge" alt="YOLO">
  <img src="https://img.shields.io/badge/Qdrant-Spatial%20Memory-DC244C?style=for-the-badge" alt="Qdrant">
</p>

<p align="center">
  <a href="https://github.com/Abhiiishek44/DrishtiSense/stargazers">
    <img src="https://img.shields.io/github/stars/Abhiiishek44/DrishtiSense?style=flat-square" alt="GitHub Stars">
  </a>
  <a href="https://github.com/Abhiiishek44/DrishtiSense/issues">
    <img src="https://img.shields.io/github/issues/Abhiiishek44/DrishtiSense?style=flat-square" alt="GitHub Issues">
  </a>
  <a href="https://github.com/Abhiiishek44/DrishtiSense">
    <img src="https://img.shields.io/github/last-commit/Abhiiishek44/DrishtiSense?style=flat-square" alt="Last Commit">
  </a>
</p>

**DrishtiSense** is a real-time spatial intelligence system for assistive navigation. It combines object detection, depth estimation, visual odometry, persistent spatial memory, obstacle awareness, and voice interaction to maintain useful context about the user's surroundings beyond the current camera frame.

The system is designed to answer spatial queries such as:

```text
"Where is my bottle?"
"What's in front of me?"
"Take me to the door."
```

Instead of treating each video frame independently, DrishtiSense maintains a continuously updated representation of detected objects, their estimated positions, and their last known state.

---

## Core Capabilities

- Real-time object detection using YOLOv8
- Multi-object tracking with temporal depth smoothing
- Monocular depth estimation using MiDaS
- 3D camera-relative object positioning
- Persistent spatial memory backed by Qdrant
- Confidence decay for stale observations
- Visual Re-Identification for object deduplication
- ORB-based visual odometry for heading estimation
- Bird's-Eye View occupancy mapping
- Local obstacle-aware navigation
- Open-vocabulary focused object search
- Voice and natural-language queries
- Event-driven multi-agent coordination
- Real-time WebSocket updates

---

## How It Works

```text
Camera
  │
  ▼
Object Detection
  │
  ▼
Tracking ───────────────┐
  │                     │
  ▼                     ▼
Depth Estimation   Visual Odometry
  │                     │
  └──────────┬──────────┘
             ▼
      3D Spatial State
             │
      ┌──────┴──────┐
      ▼             ▼
 World Model   Spatial Memory
      │             │
      └──────┬──────┘
             ▼
      Navigation Layer
             │
             ▼
      Safety Validation
             │
             ▼
       Voice Response
```

A detected object is associated with depth, direction, tracking state, and spatial metadata. When it leaves the camera view, the last valid observation remains available in spatial memory and can be queried later.

---

## Runtime Architecture

DrishtiSense separates latency-sensitive perception from slower reasoning tasks.

### Perception Loop

The perception loop handles:

```text
camera frame
→ visual odometry
→ object detection
→ object tracking
→ depth estimation
→ 3D projection
→ occupancy update
→ safety checks
```

This path does not depend on LLM latency.

### Event-Driven Agent Layer

Runtime coordination is handled through an asynchronous Pub/Sub event bus.

```text
vision/new_frame
      ↓
ArchivistAgent
      ↓
memory/candidates_ready
      ↓
JanitorAgent
      ↓
memory/write_approved
      ↓
Spatial Memory
```

User queries follow a separate path:

```text
system/query_received
      ↓
LibrarianAgent
      ↓
memory/search_result
      ↓
CoordinatorAgent
      ↓
navigation/route_proposed
      ↓
CriticAgent
      ↓
navigation/route_final
```

Agents communicate through events rather than direct inter-agent calls.

---

## Agent Responsibilities

| Agent | Responsibility |
|---|---|
| `ArchivistAgent` | Converts tracked detections into spatial-memory candidates |
| `JanitorAgent` | Deduplicates observations using Re-ID, tracking, and spatial proximity |
| `LibrarianAgent` | Retrieves stored objects and evaluates memory confidence |
| `CoordinatorAgent` | Resolves user queries and generates navigation instructions |
| `CriticAgent` | Validates routes against confidence and obstacle state |
| `AvoiderAgent` | Produces local obstacle-avoidance guidance |

---

## Spatial Memory

DrishtiSense stores more than object labels.

A spatial-memory entry includes:

```python
SpatialMemory:
    label
    confidence
    original_confidence

    distance_m
    angle_abs

    translation_x
    translation_y
    translation_z
    azimuth_deg

    reid_embedding

    timestamp
    session_id
    user_id
```

Memory confidence decays gradually over time instead of using a hard expiration threshold.

This allows the system to distinguish between:

```text
currently visible
recently observed
stale memory
low-confidence memory
```

---

## Object Re-Identification

Tracking IDs are temporary and cannot reliably identify an object after it disappears and later returns.

DrishtiSense combines:

- LAB chroma features
- Local Binary Pattern texture
- spatial color distribution
- spatial proximity
- recent track history

to reduce duplicate memories and improve cross-frame object association.

---

## Depth and 3D Positioning

MiDaS provides monocular relative depth.

DrishtiSense applies additional calibration and smoothing before using depth in navigation:

```text
relative depth
    ↓
multi-object anchors
    ↓
RANSAC calibration
    ↓
scale estimate
    ↓
Kalman smoothing
    ↓
estimated metric distance
```

Object location is then represented using camera-relative 3D coordinates:

```python
translation_x  # left / right
translation_y  # vertical offset
translation_z  # forward depth
azimuth_deg    # relative horizontal angle
```

---

## Visual Odometry

Camera heading is estimated using an ORB-based visual odometry pipeline:

```text
ORB feature extraction
        ↓
FLANN feature matching
        ↓
Lowe ratio filtering
        ↓
RANSAC Essential Matrix
        ↓
recoverPose()
        ↓
relative heading
```

For mobile deployments, external metric pose data can be supplied through the camera-pose API.

---

## Local Obstacle Awareness

Detected obstacles are projected into a Bird's-Eye View occupancy grid.

The grid tracks:

```text
FREE
OCCUPIED
UNKNOWN
```

The navigation layer uses this representation before suggesting lateral movement.

Example:

```text
Target direction: straight
Obstacle: center
Available space: left

→ "Move slightly left, then continue forward."
```

---

## Focused Object Search

Continuous scene detection remains lightweight.

For explicit user searches, DrishtiSense supports a focused detection path:

```text
target query
    ↓
YOLO-World
    ↓
Grounding DINO
    ↓
optional SAM2 refinement
    ↓
temporal tracking
```

This is useful for objects that are not well represented in the standard detector.

---

## API

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Application health |
| `GET` | `/status` | Runtime component status |
| `GET` | `/memory` | Spatial memory snapshot |
| `GET` | `/scene` | Current world-model state |
| `GET` | `/find-object/{label}` | Find a visible or remembered object |
| `GET` | `/safe-path` | Local free-space estimate |
| `POST` | `/camera-pose` | External ARCore/VIO pose input |

Example:

```http
GET /find-object/bottle
```

```json
{
  "object": "bottle",
  "visible": false,
  "distance": 2.1,
  "direction": "behind-right",
  "last_seen": "12 seconds ago"
}
```

### WebSocket

```text
/ws
```

Client query:

```json
{
  "type": "query",
  "text": "where is my bottle?"
}
```

Supported real-time updates include:

- frames
- detections
- navigation responses
- safety alerts
- memory updates
- world-model updates
- system status
- agent events

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| Computer Vision | OpenCV |
| Object Detection | YOLOv8 |
| Open-Vocabulary Detection | YOLO-World |
| Focused Detection | Grounding DINO |
| Segmentation | SAM2 (optional) |
| Depth Estimation | MiDaS |
| Tracking | IoU Tracker + Kalman Filter |
| Visual Odometry | ORB, FLANN, Essential Matrix |
| Spatial Memory | Qdrant |
| Embeddings | Sentence Transformers |
| Agent Runtime | Python `asyncio` |
| Event System | Pub/Sub EventBus |
| LLM Providers | Groq, OpenAI |
| Local LLM | llama.cpp / Ollama |
| Validation | Pydantic |

---

## Project Structure

```text
DrishtiSense/
├── main.py
├── orchestrator.py
├── agents.py
├── event_bus.py
├── vision.py
├── models.py
├── .env.example
└── README.md
```

### `main.py`

FastAPI application, REST endpoints, WebSocket handling, configuration, and runtime state.

### `orchestrator.py`

Initializes the perception pipeline, agents, memory layer, and event subscriptions.

### `agents.py`

Contains the event-driven memory, coordination, validation, and avoidance agents.

### `event_bus.py`

Async Pub/Sub event infrastructure and message definitions.

### `vision.py`

Contains camera management, detection, tracking, depth estimation, Re-ID, occupancy mapping, and visual odometry.

### `models.py`

Shared spatial data models and geometry utilities.

---

## Installation

### Requirements

- Python 3.10+
- Qdrant
- Webcam or IP camera
- CUDA-capable GPU optional

### Clone

```bash
git clone https://github.com/Abhiiishek44/DrishtiSense.git
cd DrishtiSense
```

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install \
  fastapi \
  "uvicorn[standard]" \
  pydantic \
  pydantic-settings \
  opencv-python-headless \
  ultralytics \
  qdrant-client \
  sentence-transformers \
  groq \
  openai
```

Optional depth runtime:

```bash
pip install torch torchvision
```

or:

```bash
pip install onnxruntime
```

### Start Qdrant

```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

### Configure Environment

```bash
cp .env.example .env
```

Example configuration:

```env
GROQ_API_KEY=
OPENAI_API_KEY=

CAMERA_MODE=local
CAMERA_INDEX=0

YOLO_MODEL=yolov8n.pt
DETECTION_CONFIDENCE=0.50

QDRANT_HOST=localhost
QDRANT_PORT=6333

MEMORY_DECAY_HALF_LIFE_HOURS=2.0
CROSS_SESSION_ENABLED=true

SAFETY_CRITICAL_DIST=0.8
SAFETY_WARNING_DIST=1.5
SAFETY_CAUTION_DIST=2.5
```

### Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Phone Camera

A phone can be used as the camera source through an MJPEG or RTSP stream.

```env
CAMERA_MODE=ip
CAMERA_IP_URL=http://<phone-ip>:8080/video
```

---

## Current Scope

DrishtiSense is currently an experimental prototype focused on:

- indoor spatial awareness
- object memory
- target finding
- local navigation guidance
- obstacle awareness
- voice interaction

The project is not intended to replace certified mobility aids or orientation-and-mobility support.

---

## License

See [`LICENSE`](LICENSE).
