# DrishtiSense

<p align="center">
  <strong>Spatial intelligence for safer, more independent mobility.</strong>
</p>

<p align="center">
  <a href="https://github.com/Abhiiishek44/DrishtiSense/stargazers">
    <img src="https://img.shields.io/github/stars/Abhiiishek44/DrishtiSense?style=flat-square" alt="GitHub Stars">
  </a>
  <a href="https://github.com/Abhiiishek44/DrishtiSense/issues">
    <img src="https://img.shields.io/github/issues/Abhiiishek44/DrishtiSense?style=flat-square" alt="GitHub Issues">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-Async-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Status-Experimental-orange?style=flat-square" alt="Status">
</p>

---

## What is DrishtiSense?

**DrishtiSense** is an open-source spatial AI system built to help visually impaired users understand, remember, and navigate their physical surroundings through computer vision and natural voice interaction.

Most camera-based assistants answer a simple question:

> **“What can the camera see right now?”**

DrishtiSense explores a harder one:

> **“What do I know about the space around the user, what has changed, and what should the user do next?”**

The system combines real-time perception, depth estimation, object tracking, spatial memory, visual odometry, safety-aware navigation, and a multi-agent reasoning layer to turn camera observations into useful guidance.

Instead of only reporting:

```text
Bottle detected
```

DrishtiSense aims to support interactions such as:

```text
User: Where is my bottle?

DrishtiSense:
I last saw your bottle near the table behind you.
Turn slightly right and I can guide you toward it.
```

The project is designed around one central idea:

> **Perception should not disappear when an object leaves the camera frame.**

---

## Why this matters

For a visually impaired user, recognizing an object is only part of the problem.

Useful environmental awareness also requires understanding:

- **what** is nearby,
- **where** it is,
- **how far away** it is,
- **whether it is moving**,
- **whether it blocks the user's path**,
- **where it was last seen**,
- **whether the environment has changed**, and
- **what action the user should take next**.

Traditional object detectors are usually frame-oriented:

```text
Camera → Detect → Report → Forget
```

DrishtiSense instead works toward persistent spatial understanding:

```text
Perceive → Track → Understand → Remember → Validate → Guide
```

That shift—from isolated detections to a continuously updated world model—is the foundation of the project.

---

## Core experience

### 1. See

Continuously perceive the environment using object detection, tracking, depth estimation, and spatial projection.

```text
Chair
0.9 m
Ahead-right
Visible now
```

### 2. Remember

Retain useful spatial information after an object leaves the current camera view.

```text
Bottle
Last seen near the table
2.1 m
Behind-right
Memory confidence: good
```

### 3. Navigate

Turn spatial information into simple instructions rather than exposing raw telemetry.

```text
↻ Turn right

↑ Walk straight · 1.8 m

↖ Move slightly left

STOP · Obstacle ahead

✓ Target reached
```

---

## Demo scenario

```text
1. Camera detects a bottle
        ↓
2. Bottle position is stored in spatial memory
        ↓
3. User turns away and the bottle leaves the camera view
        ↓
4. User asks: "Where is my bottle?"
        ↓
5. DrishtiSense resolves the remembered target
        ↓
6. User says: "Take me there"
        ↓
7. Navigation guidance begins
        ↓
8. A chair appears in the route
        ↓
9. Safety layer interrupts the route
        ↓
10. User is guided around the obstacle
        ↓
11. Bottle is visually reacquired
        ↓
12. Target position is corrected and navigation completes
```

The goal is not to build another object-detection dashboard.

The goal is to build a system that can maintain **spatial continuity** while the user and camera move.

---

# Architecture

DrishtiSense uses an asynchronous, event-driven architecture.

Latency-sensitive perception and safety logic are kept separate from slower reasoning and language-model operations.

```text
                         ┌────────────────────┐
                         │      Camera        │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │    Perception      │
                         │ Detection + Depth  │
                         │ Tracking + Pose    │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │    World Model     │
                         └─────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
             Spatial Memory                 Safety Cortex
                    │                             │
                    ▼                             │
              Agent System                       │
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                           Navigation Engine
                                   │
                                   ▼
                            Voice Guidance
```

---

## Fast loop vs. cognitive loop

### Fast loop

```text
Camera Capture
      ↓
Visual Odometry
      ↓
Object Detection
      ↓
Object Tracking
      ↓
Depth Estimation
      ↓
3D Projection
      ↓
Occupancy Update
      ↓
Safety Evaluation
```

This path handles information that should not wait for an LLM.

### Cognitive loop

```text
Spatial Memory
      ↓
Retrieval
      ↓
Goal Resolution
      ↓
Route Proposal
      ↓
Route Validation
      ↓
Natural-Language Guidance
```

> **A slow language-model call must never delay a safety-critical warning.**

---

# Multi-Agent System

DrishtiSense uses specialized agents with explicit responsibilities and event-driven communication.

| Agent | Responsibility |
|---|---|
| **Archivist** | Converts current perception into candidate spatial memories |
| **Janitor** | Deduplicates observations using tracking, Re-ID, and spatial proximity |
| **Librarian** | Retrieves stored objects and evaluates memory confidence |
| **Coordinator** | Resolves user intent and proposes navigation actions |
| **Critic** | Validates proposed guidance against safety and memory reliability |
| **Avoider** | Produces local obstacle-avoidance guidance |

Agents communicate through a Pub/Sub event bus rather than tightly calling one another.

This provides:

- loose coupling,
- asynchronous execution,
- fault isolation,
- observable event flow,
- independent route validation,
- priority handling for safety events.

---

# Spatial intelligence pipeline

```text
Camera Frame
     │
     ▼
┌───────────────┐
│     YOLO      │
│   Detection   │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Object Track  │
└───────┬───────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
 Depth     Visual
 Engine    Odometry
   │         │
   └────┬────┘
        ▼
  3D Projection
        │
   ┌────┴─────────┐
   │              │
   ▼              ▼
World Model   Spatial Memory
   │              │
   └──────┬───────┘
          ▼
    Goal Resolution
          │
          ▼
     Path Guidance
          │
          ▼
    Safety Validation
          │
          ▼
     Voice Response
```

---

# Technical capabilities

## Real-time object perception

DrishtiSense uses real-time object detection with multi-object tracking to maintain continuity across frames.

The system can maintain:

- object label,
- detection confidence,
- tracking identity,
- bounding box,
- estimated depth,
- 3D camera-relative position,
- direction / azimuth.

The user-facing interface intentionally converts these values into human guidance instead of exposing raw model outputs.

---

## Monocular depth estimation

A monocular depth model can estimate relative scene depth from a single camera, but relative depth does not automatically equal accurate metric distance.

DrishtiSense improves distance stability using:

- multi-anchor calibration,
- RANSAC consensus,
- temporal smoothing,
- per-track filtering,
- tracked-object history.

The purpose is to reduce frame-to-frame distance jumps and produce more stable navigation information.

---

## 3D spatial projection

```text
2D Detection
    ↓
Depth Estimate
    ↓
Camera Projection
    ↓
(X, Y, Z)
    ↓
Relative Distance + Bearing
```

These spatial representations are used by memory and navigation rather than relying on a language model to estimate geometry.

---

## Persistent spatial memory

A spatial memory can include:

```python
SpatialMemory:
    label
    confidence
    original_confidence

    translation_x
    translation_y
    translation_z

    distance_m
    direction
    azimuth_deg

    reid_embedding

    timestamp
    session_id
    user_id
```

This enables the system to answer queries about recently observed objects even when they are not currently visible.

---

## Confidence-aware memory

```text
Fresh observation
      ↓
High confidence
      ↓
Good confidence
      ↓
Moderate confidence
      ↓
Low-confidence memory
      ↓
Request better perception
```

This allows the system to respond naturally to stale information instead of treating every stored memory as certain.

---

## Visual Re-Identification

DrishtiSense uses appearance and spatial signals to help associate new detections with existing memories.

The Re-ID representation combines:

- LAB chroma information,
- Local Binary Pattern texture,
- spatial color layout,
- spatial proximity,
- recent tracking history.

This helps reduce duplicate memories for the same physical object.

---

## Visual odometry

The desktop pipeline uses ORB-based visual odometry:

```text
ORB Features
     ↓
Feature Matching
     ↓
RANSAC Essential Matrix
     ↓
Camera Pose Recovery
     ↓
Relative Heading
```

For mobile deployment, metric pose information can be supplied through ARCore or another Visual-Inertial Odometry source.

---

## Bird's-Eye occupancy representation

```text
Top-down view

        obstacle
          ███

      █████████
      █ clear █
      █████████

         USER
```

The navigation layer can query this representation before suggesting lateral motion.

---

## Goal-based navigation

```text
"Take me to my bottle"
          │
          ▼
     Resolve Target
          │
    ┌─────┴─────┐
    │           │
 Visible     Remembered
    │           │
    └─────┬─────┘
          ▼
   Target Position
          │
          ▼
 Navigation Vector
          │
          ▼
 Safety Validation
          │
          ▼
   Voice Guidance
```

Distance, heading, and direction come from the spatial pipeline.

The LLM is responsible for **communication**, not physical measurement.

---

## Active perception

```text
Object Query
    ↓
Memory Found
    ↓
Confidence Too Low
    ↓
Request New Observation
    ↓
Re-detect
    ↓
Update Memory
```

The system can request more visual evidence rather than confidently responding from weak memory.

---

## Focused object search

```text
User Target
    ↓
YOLO-World
    ↓
Grounding DINO
    ↓
Optional SAM2 Refinement
    ↓
Temporal Tracking
```

This creates two perception modes:

1. continuous environmental awareness,
2. targeted object search.

---

# Safety model

Safety events take priority over normal navigation.

```text
Target straight ahead
        ↑

      Chair
       ⚠

      User
```

Instead of continuing to say:

> “Walk straight.”

the safety layer can interrupt:

> **“Stop. A chair is blocking your path. Move slightly left.”**

Navigation and safety are treated as separate problems:

```text
Goal Direction
      +
Obstacle Map
      ↓
Safe Immediate Action
```

---

# API

## REST endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health and active connections |
| `GET` | `/status` | Runtime component status |
| `GET` | `/memory` | Spatial-memory snapshot |
| `GET` | `/scene` | Current world-model state |
| `GET` | `/find-object/{label}` | Locate a visible or remembered object |
| `GET` | `/safe-path` | Lightweight local free-space estimate |
| `POST` | `/camera-pose` | Supply metric camera/VIO pose |

Example:

```http
GET /find-object/bottle
```

Possible response:

```json
{
  "object": "bottle",
  "visible": false,
  "distance_m": 2.1,
  "direction": "behind-right",
  "last_seen": "12 seconds ago"
}
```

---

## WebSocket

```text
/ws
```

The real-time channel can stream:

- camera frames,
- detections,
- navigation responses,
- safety alerts,
- memory updates,
- world-model events,
- agent activity,
- active-perception requests.

Example:

```json
{
  "type": "query",
  "text": "where is my bottle?"
}
```

---

# Technology stack

| Layer | Technology |
|---|---|
| **API** | FastAPI, Uvicorn |
| **Computer Vision** | OpenCV, YOLOv8 |
| **Open-Vocabulary Detection** | YOLO-World |
| **Focused Detection** | Grounding DINO |
| **Segmentation** | Optional SAM2 |
| **Depth Estimation** | MiDaS |
| **Tracking** | IoU + Kalman filtering |
| **Visual Odometry** | ORB, FLANN, Essential Matrix |
| **Spatial Memory** | Qdrant |
| **Embeddings** | Sentence Transformers |
| **Agent Runtime** | Python asyncio + EventBus |
| **Cloud LLM** | Groq / OpenAI |
| **Edge LLM** | llama.cpp / Ollama |
| **Validation** | Pydantic |

---

# Quick start

## Prerequisites

- Python 3.10+
- Qdrant
- Webcam or IP camera
- Optional CUDA-capable GPU

## Clone

```bash
git clone https://github.com/Abhiiishek44/DrishtiSense.git
cd DrishtiSense
```

## Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

## Install dependencies

```bash
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings \
  opencv-python-headless ultralytics qdrant-client \
  sentence-transformers groq openai
```

Optional depth runtime:

```bash
pip install torch torchvision
```

or:

```bash
pip install onnxruntime
```

## Start Qdrant

```bash
docker run --name drishtisense-qdrant -p 6333:6333 qdrant/qdrant
```

## Configure

```bash
cp .env.example .env
```

Example:

```env
GROQ_API_KEY=
OPENAI_API_KEY=

CAMERA_MODE=local
CAMERA_INDEX=0

YOLO_MODEL=yolov8n.pt
DETECTION_CONFIDENCE=0.50

QDRANT_HOST=localhost
QDRANT_PORT=6333
CROSS_SESSION_ENABLED=true

SAFETY_CRITICAL_DIST=0.8
SAFETY_WARNING_DIST=1.5
SAFETY_CAUTION_DIST=2.5
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

# Phone camera

```env
CAMERA_MODE=ip
CAMERA_IP_URL=http://192.168.x.x:8080/video
```

This allows the backend to run on a laptop or workstation while the phone acts as the visual sensor.

---

# Repository structure

```text
DrishtiSense/
│
├── main.py
│   └── FastAPI app, runtime configuration, REST/WebSocket APIs
│
├── orchestrator.py
│   └── Component wiring and perception-loop bootstrap
│
├── agents.py
│   └── Spatial-memory and navigation agents
│
├── event_bus.py
│   └── Async Pub/Sub event infrastructure
│
├── vision.py
│   ├── object detection
│   ├── tracking
│   ├── depth estimation
│   ├── Re-ID
│   ├── occupancy mapping
│   └── visual odometry
│
├── models.py
│   └── Spatial data models and geometry utilities
│
└── .env.example
    └── Runtime configuration template
```

---

# Design principles

## Geometry before language

Distance, direction, heading, object position, and arrival should come from deterministic spatial computation.

The LLM communicates those values. It does not invent them.

## Safety before reasoning

Immediate hazard handling belongs in the low-latency perception path.

```text
Safety Loop > LLM Loop
```

## Memory should express uncertainty

A five-second-old observation and a two-hour-old observation should not be presented with the same certainty.

## Navigation should produce actions

Avoid:

```text
azimuth=-21.6
translation_z=1.43
confidence=0.72
```

Prefer:

> **“Move slightly left. The target is approximately 1.4 metres ahead.”**

## Agents should have a reason to exist

Agents own independent responsibilities such as memory creation, deduplication, retrieval, route generation, route validation, and obstacle avoidance.

---

# Project status

> [!IMPORTANT]
> **DrishtiSense is an experimental assistive spatial-intelligence prototype.**

It is intended for research, experimentation, accessibility prototyping, hackathons, and spatial-AI development.

It is **not currently a replacement** for mobility canes, guide dogs, trained orientation-and-mobility assistance, or certified assistive navigation devices.

Object detection, monocular depth, tracking, remembered positions, and navigation guidance can be incorrect.

Real-world deployment requires extensive accessibility testing, sensor redundancy, hardware validation, fail-safe behavior, and evaluation with the people the system is intended to support.

---

# Roadmap

- [ ] Metric Visual-Inertial Odometry
- [ ] ARCore / ARKit world anchoring
- [ ] Persistent room-scale mapping
- [ ] Cross-room spatial memory
- [ ] Semantic room understanding
- [ ] Free-space semantic segmentation
- [ ] Dynamic obstacle trajectory prediction
- [ ] Haptic guidance
- [ ] Smart-glasses integration
- [ ] Fully on-device inference
- [ ] Outdoor navigation
- [ ] Accessibility-focused user studies
- [ ] Benchmarking for distance and navigation accuracy

---

# Research direction

Most computer-vision systems answer:

> **What is visible?**

DrishtiSense is exploring a broader problem:

> **What is around the user, what has changed, what should be remembered, and what action should happen next?**

That is the transition from object detection toward **persistent spatial intelligence**.

---

# Contributing

Contributions are welcome.

Areas where open-source collaboration can create meaningful improvements include:

- computer vision,
- SLAM / VIO,
- accessibility,
- robotics,
- depth estimation,
- object Re-ID,
- multimodal AI,
- spatial computing,
- edge inference,
- navigation,
- multi-agent systems.

For major architectural changes, please open an issue first so the problem, safety implications, and implementation direction can be discussed before development.

---

# Responsible use

DrishtiSense works in a safety-sensitive domain.

Changes that affect navigation, obstacle avoidance, distance estimation, emergency warnings, or route generation should include a clear explanation of failure modes and testing assumptions.

Do not treat model confidence as physical certainty.

---

# Acknowledgements

DrishtiSense builds on the open-source AI and computer-vision ecosystem, including:

- Ultralytics
- OpenCV
- MiDaS
- Qdrant
- Grounding DINO
- SAM2
- FastAPI
- Sentence Transformers
- llama.cpp
- Ollama

---

<p align="center">
  <strong>Perceive the world. Remember the space. Navigate with confidence.</strong>
</p>

<p align="center">
  <sub>Built to explore how spatial AI can make environmental understanding more useful, continuous, and accessible.</sub>
</p>
