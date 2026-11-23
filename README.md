# Helmet Detection Using YOLOv8

Real-time motorcycle rider helmet detection (`With Helmet` vs `Without Helmet`) built with **YOLOv8**, **Python**, and **OpenCV**, featuring spatial motorcycle filtering to ignore walking pedestrians and live webcam support.

## Model Performance (YOLOv8n)
- **Overall mAP@50:** 90.7%
- **Precision:** 83.4% | **Recall:** 86.3%
- **With Helmet mAP@50:** 93.9%
- **Without Helmet mAP@50:** 87.5%

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run on traffic video: `python detect_video.py --source sample_videos/traffic.mp4 --conf 0.45`
3. Run live webcam (roadside): `python detect_video.py --source 0`
4. Run live webcam (indoor demo): `python detect_video.py --source 0 --no-moto-filter`

## Acknowledgements
Special thanks to trainer **Upender Muthyala** for guidance and support throughout this project.

## ?? Demo Output
- [Watch / Download Annotated Detection Video](outputs/helmet_detected.mp4)
