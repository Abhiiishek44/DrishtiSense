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
DrishtiSense is an assistive spatial intelligence system that builds a persistent understanding of the surrounding environment from a live camera feed.

By combining real-time vision, depth estimation, visual odometry, spatial memory, and obstacle-aware navigation, it can track objects beyond the current frame, reason about their relative position, and provide context-aware guidance as the user moves.

Perception that persists beyond the camera frame.
---

## Core Capabilities

- Real-time object detection with YOLOv8
- Multi-object tracking and temporal depth smoothing
- Monocular depth estimation with MiDaS
- 3D camera-relative object positioning
- Persistent spatial memory with Qdrant
- Confidence decay for stale observations
- Visual Re-Identification for memory deduplication
- ORB-based visual odometry
- Bird's-Eye View occupancy mapping
- Local obstacle-aware navigation
- Focused open-vocabulary object search
- Voice and natural-language queries
- Event-driven multi-agent coordination
- Real-time WebSocket updates

---

## Architecture

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

DrishtiSense separates latency-sensitive perception from slower memory and reasoning tasks.

### Perception Loop

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

### Agent Layer

Runtime coordination is handled through an asynchronous Pub/Sub event bus.

```text
vision/new_frame
      ↓
ArchivistAgent
      ↓
JanitorAgent
      ↓
Spatial Memory
```

User queries follow:

```text
system/query_received
      ↓
LibrarianAgent
      ↓
CoordinatorAgent
      ↓
CriticAgent
      ↓
navigation/route_final
```

| Agent | Responsibility |
|---|---|
| `ArchivistAgent` | Converts tracked detections into memory candidates |
| `JanitorAgent` | Deduplicates observations |
| `LibrarianAgent` | Retrieves spatial memories |
| `CoordinatorAgent` | Resolves queries and proposes navigation |
| `CriticAgent` | Validates route confidence and safety |
| `AvoiderAgent` | Produces local obstacle-avoidance guidance |

---

## Spatial Memory

DrishtiSense stores spatial and temporal state for previously observed objects.

```python
SpatialMemory:
    label
    confidence
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

Memories decay gradually instead of expiring immediately, allowing the system to distinguish between recent and stale observations.

Tracking, Re-ID, spatial proximity, and recency are combined to reduce duplicate memories when the same object disappears and later returns.

---

## Navigation

Navigation is separated into two stages:

```text
Current Pose + Target Position
              ↓
     Target Direction
              +
      Occupancy State
              ↓
    Safe Immediate Action
```

Typical navigation states include:

```text
TURN_LEFT
TURN_RIGHT
ALIGNED
WALK_FORWARD
MOVE_LEFT
MOVE_RIGHT
STOP
TARGET_REACHED
TARGET_UNCERTAIN
```

Distance, heading, and obstacle clearance are produced by the spatial pipeline. The language model is used only to convert verified system state into natural guidance.

---

## Focused Object Search

Continuous perception uses lightweight detection.

For explicit object searches, DrishtiSense can optionally use:

```text
YOLO-World
    ↓
Grounding DINO
    ↓
SAM2 refinement
    ↓
Temporal tracking
```

This allows targeted searches for objects that are not covered reliably by the standard detector.

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

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| Computer Vision | OpenCV |
| Object Detection | YOLOv8 |
| Open-Vocabulary Detection | YOLO-World |
| Focused Detection | Grounding DINO |
| Segmentation | SAM2 |
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

DrishtiSense currently focuses on:

- indoor spatial awareness
- object memory
- target finding
- local navigation guidance
- obstacle awareness
- voice interaction

> DrishtiSense is an experimental prototype and is not a replacement for certified mobility aids or orientation-and-mobility support.

---

## License

See [`LICENSE`](LICENSE).
