# PPE Detection using YOLOv8

A YOLOv8-based Personal Protective Equipment (PPE) detection system developed to identify PPE compliance in images, videos, and live webcam input.

## Overview

This project uses a trained YOLOv8 object detection model to detect Personal Protective Equipment and identify potential safety violations.

The application provides a Streamlit-based interface for running PPE detection through different input sources.

## Features

- Image-based PPE detection
- Video-based PPE detection
- Live webcam detection
- Adjustable detection settings
- Bounding-box visualization
- Confidence scores for detected objects
- Safety violation flagging

## Detected Classes

The trained model detects the following classes:

- Helmet
- No-Helmet
- Vest
- No-Vest
- Person

Classes beginning with `NO-` or `no-` are treated as safety violations by the application.

## Technologies Used

- Python
- YOLOv8
- Ultralytics
- OpenCV
- Streamlit
- NumPy
- Pillow

## Dataset

The project uses a Personal Protective Equipment (PPE) object detection dataset containing training and testing images with corresponding annotations.

The dataset is included in the repository under:

```text
dataset/
├── train/
├── test/
└── data.yaml