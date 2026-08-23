# UIT Car Racing 2024 – Professional Category

**Third Prize | 4-member team**

Portfolio documentation of an autonomous-driving project developed for **UIT Car Racing 2024 – Professional Category**.

The project combined a YOLOv5-based perception workflow with lane-following and steering-control logic. Development progressed from the simulator used during the online stages toward preparation for a physical vehicle using real-camera input.

> This repository is a cleaned portfolio version of the original competition project.  
> It contains selected source code, historical controller experiments, a sanitized YOLOv5 training notebook, and representative simulator / real-camera samples.

---

## Overview

The system involved two main areas:

### Perception

- Image preparation and labelling
- Traffic-sign / driving-command detection using YOLOv5
- OpenCV-based image processing
- Training and evaluation using Roboflow and Google Colab
- Simulator and real-camera data

### Lane Following & Control

- Processing simulator segmentation frames
- Grayscale conversion and intensity normalisation
- Road-centre estimation at multiple image checkpoints
- Road-slope estimation
- Steering-angle generation
- PID-based control
- Kalman steering smoothing
- Dynamic speed adjustment
- Testing and tuning for sharp and 90-degree turns

---

# My Contribution

I worked primarily on the **computer-vision / perception side** of the project and also contributed to the lane-following and controller development.

My contribution included:

- Preparing and labelling image data used by the perception pipeline.
- Training a custom **YOLOv5** model for traffic-sign / driving-command recognition.
- Working with Python and OpenCV during the online autonomous-driving stages.
- Contributing to lane-following logic based on simulator segmentation frames.
- Supporting testing and tuning for difficult road sections, including sharp and 90-degree turns.
- Collaborating within a **4-member team** throughout the competition.

The competition result reflects the work of the whole team. This repository documents the parts of the project that I worked with and can explain rather than claiming sole authorship of the complete system.

---

# Project Evolution

The repository preserves several stages of controller development.

The earlier experimental scripts explored simpler approaches such as:

- single-checkpoint road-centre detection;
- geometric steering-angle calculation;
- fixed vehicle speed;
- three-checkpoint weighted-slope estimation;
- PID steering correction;
- left/right lane bias.

These experiments eventually led to the more complete controller stored at:

```text
src/controller.py
```

The main controller integrates:

- three road checkpoints;
- adaptive straight/curve PID settings;
- Kalman filtering;
- dynamic speed control;
- steering smoothing;
- additional handling for aggressive turns.

---

# Perception System

## YOLOv5 Detection

A custom YOLOv5 detector was trained using a Roboflow-formatted dataset.

The preserved dataset configuration contains **6 classes**:

| Class |
|---|
| `left` |
| `no right` |
| `noleft` |
| `right` |
| `stop` |
| `straight` |

These classes represented traffic signs or driving commands used in the autonomous-driving environment.

---

## Training Configuration

The preserved training run used:

| Parameter | Value |
|---|---:|
| Model family | YOLOv5 |
| Input size | 416 × 416 |
| Batch size | 16 |
| Epochs | 100 |
| Number of classes | 6 |
| Framework | PyTorch / Ultralytics YOLOv5 |
| Dataset workflow | Roboflow |
| Training environment | Google Colab / GPU |

The notebook used a training command equivalent to:

```bash
python train.py \
  --img 416 \
  --batch 16 \
  --epochs 100 \
  --data <dataset>/data.yaml \
  --cfg ./models/custom_yolov5s.yaml \
  --weights '' \
  --name yolov5s_results \
  --cache
```

---

## Preserved Model Results

Validation of the saved `best.pt` checkpoint produced:

| Metric | Result |
|---|---:|
| Precision | **0.751** |
| Recall | **0.900** |
| mAP@0.5 | **0.964** |
| mAP@0.5:0.95 | **0.716** |

The recorded validation set contained:

- **35 images**
- **36 annotated instances**

> These values are metrics from the preserved project training run, not official UIT Car Racing competition scores.  
> The validation sample is relatively small, so the results should not be interpreted as evidence of broad real-world generalisation.

---

# Data Samples

Two visually different sources of data were encountered during development:

1. **Simulator data** used during the earlier online competition stages.
2. **Real-camera images** collected while preparing for the physical final-round environment.

This created a practical simulator-to-real domain shift.

---

## Simulator Data

<p align="center">
  <img src="assets/simulator_01.jpg" width="31%" />
  <img src="assets/simulator_02.jpg" width="31%" />
  <img src="assets/simulator_03.jpg" width="31%" />
</p>

The simulator provided a comparatively controlled environment for developing and testing perception and driving logic.

---

## Real-Camera Data

<p align="center">
  <img src="assets/real_camera_green_light.png" width="31%" />
  <img src="assets/real_camera_left.png" width="31%" />
  <img src="assets/real_camera_parking.png" width="31%" />
</p>

These samples were collected from a real camera during preparation for the physical competition environment.

Compared with simulation, real-camera images introduced additional variation in:

- lighting;
- perspective;
- road texture;
- background objects;
- image noise;
- sign appearance and scale.

The images are included to document the transition from a controlled simulator toward real-camera perception. They do not represent a complete reproduction of the final physical vehicle system.

---

# Lane-Following Pipeline

The controller received segmentation frames from the competition environment and estimated road direction from them.

The main pipeline was:

```text
Segmentation Frame
        ↓
Grayscale Conversion
        ↓
Intensity Normalisation
        ↓
Road-Centre Sampling
at Three Checkpoints
        ↓
Road-Slope Estimation
        ↓
Steering-Angle Mapping
        ↓
Kalman Smoothing
        ↓
PID Control
        ↓
Dynamic Speed Adjustment
        ↓
Vehicle Command
```

---

## Three-Checkpoint Road Estimation

The main controller samples the road at three vertical image positions.

```python
UPPER_CHECKPOINT = 150
MID_CHECKPOINT   = 130
LOWER_CHECKPOINT = 120
```

Instead of estimating road direction from only one horizontal row, the controller uses multiple road-centre measurements to obtain a more useful indication of the road geometry.

The average slope is then mapped into a steering command.

---

## Grayscale & Intensity Normalisation

Simulator segmentation frames are converted to grayscale:

```python
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

The intensity is then normalised before road-centre extraction.

This simplifies the input used by the lane-following algorithm and allows the controller to focus on the road geometry represented by the segmentation frame.

---

# Steering Control

## Adaptive PID

The main controller contains separate PID settings for straight and curved sections:

```python
PID_STRAIGHT = (0.8, 0.01, 0.3)
PID_CURVE    = (1.4, 0.02, 0.5)
```

The controller switches between the two configurations according to the estimated road slope.

This was used instead of relying on one fixed PID configuration for every road condition.

---

## Kalman Filtering

A Kalman filter is used before the final PID steering correction.

Its role is to smooth the steering estimate and reduce abrupt changes caused by noisy or unstable measurements.

The filtered steering value is then passed into the PID controller.

---

## Dynamic Speed Control

Vehicle speed is adjusted according to steering behaviour.

The controller defines:

```python
SLOW_DOWN_ANGLE = 17
```

When the steering-angle change exceeds approximately **17°**, speed is reduced substantially.

Otherwise, vehicle speed decreases progressively as steering demand increases.

This was intended to improve stability during more aggressive turns.

---

# Development Challenges

## Sharp & 90-Degree Turns

One of the main practical problems during testing was **boundary overshoot**.

During aggressive turns, the vehicle could:

- enter the corner too quickly;
- react too late;
- apply a large steering correction;
- cross the intended road boundary.

Development therefore involved repeated adjustment of:

- steering behaviour;
- straight/curve PID settings;
- speed;
- steering smoothing;
- turn-specific behaviour.

---

## Simulator-to-Real Transition

A controller or perception model that performs well in simulation does not automatically behave the same way with physical-camera input.

The preparation stage exposed differences in:

- illumination;
- perspective;
- background clutter;
- texture;
- noise;
- camera placement;
- object scale.

The real-camera samples in this repository document that transition.

---

## Limited Perception Dataset

The preserved YOLOv5 validation run produced strong mAP values, but the validation set was small.

For that reason, the training metrics are presented as **development results** rather than claims of production-level model accuracy.

This distinction is especially important when moving from simulator conditions to real-camera data.

---

# Historical Controller Experiments

Earlier controller variants are preserved in:

```text
experiments/
```

They are kept because they show how the control approach evolved during development.

### `Test.py`

A simple baseline using:

- one road checkpoint;
- geometric steering-angle calculation;
- fixed vehicle speed.

### `UpdateMidLane.py`

An early centre-lane implementation using:

- one checkpoint;
- grayscale segmentation processing;
- geometric steering calculation.

### `UpdateRightLane.py`

A later version introducing:

- three road checkpoints;
- weighted road-slope estimation;
- PID steering correction;
- dynamic speed control.

### `UpdateLeftLane.py`

An experimental lane-selection variant including:

- left/right lane bias;
- three checkpoints;
- PID control;
- dynamic speed control.

These files are development artefacts rather than the recommended entry point.

The main portfolio controller is:

```text
src/controller.py
```

---

# Training Notebook

The sanitized YOLOv5 notebook is located at:

```text
notebooks/yolov5_training.ipynb
```

It preserves the original project workflow including:

- YOLOv5 setup;
- dataset configuration;
- model configuration;
- training commands;
- training output;
- validation output;
- inference workflow.

The workflow uses:

- Ultralytics YOLOv5
- PyTorch
- Roboflow
- Google Colab

---

## Credential Safety

The original notebook contained a private Roboflow API credential.

The public portfolio copy has been sanitized and does **not** expose the original credential.

To use the Roboflow download cell, provide your own environment variable:

```text
ROBOFLOW_API_KEY
```

For example:

```python
import os

api_key = os.environ.get("ROBOFLOW_API_KEY")
```

A valid Roboflow account and appropriate dataset access would still be required to reproduce the original download step.

---

# Repository Structure

```text
UIT-Car-Racing-2024/
│
├── README.md
│
├── src/
│   └── controller.py
│
├── experiments/
│   ├── Test.py
│   ├── UpdateLeftLane.py
│   ├── UpdateMidLane.py
│   └── UpdateRightLane.py
│
├── notebooks/
│   └── yolov5_training.ipynb
│
└── assets/
    ├── real_camera_green_light.png
    ├── real_camera_left.png
    ├── real_camera_parking.png
    ├── simulator_01.jpg
    ├── simulator_02.jpg
    └── simulator_03.jpg
```

---

# Dependencies

The controller code uses:

- Python
- OpenCV
- NumPy
- `simple-pid`

The main controller also imports the competition client interface:

```python
from client_lib import GetStatus, GetSeg, AVControl, CloseSocket
```

`client_lib` and the full competition simulator are **not included in this repository** because they belong to the competition environment.

Therefore, the controller should not be expected to run as a standalone application without the appropriate UIT Car Racing simulator/client environment.

---

# Technologies

## Computer Vision

- OpenCV
- YOLOv5
- Roboflow
- Image preprocessing
- Grayscale conversion
- Intensity normalisation

## Control

- PID control
- Adaptive PID tuning
- Kalman filtering
- Dynamic speed adjustment
- Road-slope-based steering

## Development

- Python
- Linux
- Google Colab
- PyTorch
- Ultralytics YOLOv5

---

# Competition Result

## UIT Car Racing 2024 – Professional Category

**Third Prize**

**4-member team**

The project provided practical exposure to combining:

- computer vision;
- autonomous-driving logic;
- image processing;
- vehicle control;
- simulation;
- real-camera perception;
- iterative engineering testing.

---

# Key Takeaways

This project was one of my first experiences working with a system in which perception directly influenced vehicle-control behaviour.

Key lessons included:

- preparing and labelling image data;
- training a custom object detector;
- evaluating model metrics critically;
- using OpenCV for image processing;
- deriving steering information from road geometry;
- connecting perception output to control logic;
- tuning PID behaviour;
- applying filtering to unstable control signals;
- adapting speed to steering demand;
- testing difficult corner cases;
- recognising simulator-to-real differences;
- collaborating within an engineering competition team.

More importantly, the project demonstrated that good performance in a controlled environment does not guarantee equivalent behaviour under different visual and physical conditions.

---

# Limitations

This repository is a **portfolio reconstruction** of the 2024 competition project rather than a complete archive of every competition resource.

Accordingly:

- the organiser-provided simulator is not redistributed;
- organiser-provided maps are not redistributed;
- the competition `client_lib` is not included;
- the complete internal team dataset is not public;
- only representative simulator and real-camera samples are provided;
- private credentials have been removed;
- some original team resources are no longer available;
- preserved YOLO metrics should not be treated as official competition results.

The objective of this repository is to document the technical work that can be responsibly and accurately presented as part of my engineering portfolio.

---

# Acknowledgements

UIT Car Racing was a **team project**, and the Third Prize result reflects the combined work of all four team members.

The project also relied on open-source tools including:

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)
- [Roboflow](https://roboflow.com/)

---

# Author

**Diep Khai Hoang**

Computer Engineering Undergraduate  
University of Information Technology  
Vietnam National University Ho Chi Minh City (**UIT-VNU-HCM**)

Interests:

- Embedded Systems
- Computer Vision
- Autonomous Systems
- Hardware–Software Integration
- Computer Systems Engineering

---

> Maintained for academic and portfolio purposes.
