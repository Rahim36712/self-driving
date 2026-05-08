# Self-Driving Car Simulation — DIP Project (Spring 2026)

A real-time self-driving car simulation built with **Python**, **OpenCV**, and **YOLOv8**. The system processes dashcam video to detect lane markings and obstacles, then outputs driving decisions (FORWARD / STOP / TURN LEFT / TURN RIGHT).

> **No real car or hardware needed** — runs entirely on a laptop with pre-recorded road video.

---

## Features

- **Lane Detection** — Classical CV pipeline: Grayscale → Gaussian Blur → Canny Edge Detection → Hough Line Transform
- **Adaptive Thresholding** — Alternative to Canny for uneven lighting (toggle with `A` key)
- **Polynomial Fitting** — Curved lane detection using `np.polyfit()` (toggle with `F` key)
- **YOLOv8 Object Detection** — Pre-trained nano model detects people, cars, bikes in real-time
- **Distance Classification** — Bounding box area → CLOSE (red) / NEAR (orange) / FAR (green)
- **Zone Detection** — Frame split into LEFT / CENTER / RIGHT to determine obstacle position
- **Rule-Based Decision Engine** — Combines lane + obstacle info → single driving command
- **Multi-Threaded Pipeline** — Lane detection + YOLO run concurrently for 15+ FPS
- **6-Panel Debug View** — Original, Edges, Lanes, YOLO, Merged, Decision (toggle with `D` key)
- **Performance Metrics** — FPS tracking, decision distribution, lane success rate

---

## Project Structure

```
dip/
├── main.py                 # Integration and Testing (Ahsan Javed)
├── lane_detection.py       # Classical lane detection (Ahsan Javed)
├── object_detection.py     # YOLOv8 obstacle detection (Rahim, Naeem, Abdullah)
├── decision.py             # Rule-based decision engine (Abdullah Khan)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── dataset/                # Road video files (.mp4)
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/Rahim36712/self-driving.git
cd self-driving

# Install dependencies
pip install -r requirements.txt
```

### Dataset
Place road video files (`.mp4`) in the `dataset/` folder.

---

## Usage

### Run the full simulation
```bash
python main.py
```

### Keyboard Controls
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `D` | Toggle 6-panel debug view |
| `P` | Pause / Resume |
| `N` | Next video (cycles through dataset) |
### Test individual modules
```bash
python lane_detection.py      # Test lane detection only
python object_detection.py    # Test YOLO detection only
python decision.py            # Test decision logic
```

---

## System Architecture

```
Video Frame
    │
    ├──► Thread 1: Lane Detection
    │    Grayscale → Blur → Canny → ROI → Hough → Lane Lines
    │
    ├──► Thread 2: YOLO Detection
    │    YOLOv8n → Bounding Boxes → Distance → Zone
    │
    └──► Main Thread: Merge + Decision
         Lane info + Obstacle info → Decision Engine → Display
```

---

## Decision Rules

| Condition | Output |
|-----------|--------|
| Close obstacle in CENTER | **STOP** |
| Close obstacle on LEFT | **TURN RIGHT** |
| Close obstacle on RIGHT | **TURN LEFT** |
| No lanes detected | **STOP - NO LANES** |
| Left lane missing | **TURN LEFT** |
| Right lane missing | **TURN RIGHT** |
| Offset > 80px right | **STEER LEFT** |
| Offset > 80px left | **STEER RIGHT** |
| All clear | **FORWARD** |

---

## Technologies Used

- **OpenCV** — Image processing (Canny, Hough, morphology)
- **NumPy** — Numerical operations, polynomial fitting
- **YOLOv8 (ultralytics)** — Real-time object detection
- **Python threading** — Concurrent pipeline execution

---

## Team

| Member | Role | File |
|--------|------|------|
| RAHIM JAMIL | YOLO Detection,Integration | `object_detection.py`,`main.py` |
| NAEEMULLAH AZIZ | YOLO Detection | `object_detection.py` |
| AHSAN JAVED | Lane Detection, Testing | `lane_detection.py`, `main.py` |
| ABDULLAH KHAN | Report, Decision | `decision.py` |

---

## License

This project is for educational purposes — Digital Image Processing course, Spring 2026.
