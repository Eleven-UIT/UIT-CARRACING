# UIT Car Racing 2024 – Professional Category

**Third Prize | 4-member team**

Autonomous-driving project developed for **UIT Car Racing 2024 – Professional Category**.  
The system combined computer-vision-based perception with lane-following and steering control for the online autonomous-driving rounds.

## My Contribution

- Prepared and labelled image data for the perception pipeline.
- Trained a YOLOv5-based perception model for the online autonomous-driving rounds.
- Contributed to the lane-following controller using Python and OpenCV.
- Supported testing and tuning for sharp and 90-degree turns.

## YOLOv5 Perception Model

The training dataset contained six traffic-sign / driving-command classes:

`left` · `no right` · `noleft` · `right` · `stop` · `straight`

Training configuration from the preserved project notebook:

- **Model:** YOLOv5
- **Epochs:** 100
- **Image size:** 416 × 416
- **Batch size:** 16
- **Precision:** 0.751
- **Recall:** 0.900
- **mAP@0.5:** 0.964
- **mAP@0.5:0.95:** 0.716

> Metrics above correspond to the preserved training run and are not an official competition benchmark.

## Lane-Following & Control Pipeline

The controller processed simulator segmentation frames through:

**Segmentation frame → grayscale conversion → intensity normalization → road-centre sampling at three checkpoints → road-slope estimation → steering command**

The control system also used:

- Separate PID settings for straight and curved sections
- Kalman filtering for steering smoothing
- Dynamic speed reduction when steering changes exceeded 17°
- Additional tuning for sharp and 90-degree turns

## Repository Contents

- Python scripts used for vehicle control and testing
- Sanitized YOLOv5 training notebook
- Project documentation and selected training artefacts

## Training Notebook

`UIT_CarRacing_YOLOv5_Training_SANITIZED.ipynb`

The original training workflow used Roboflow and Ultralytics YOLOv5.  
The original Roboflow API credential has been removed from the public portfolio copy.

To rerun the dataset-download cell, provide your own API key through the `ROBOFLOW_API_KEY` environment variable.

## Notes

This repository is maintained as a portfolio version of the 2024 competition project.  
Some competition resources, simulator assets and internal team data are not redistributed here.
