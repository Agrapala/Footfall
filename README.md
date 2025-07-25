# Footfall Human Analytics System

## Overview
This project is a modular, real-time human analytics system for:
- Age and gender detection (multi-person, robust)
- Unique human counting
- Human motion and activity analysis
- Face recognition for security/identification
- Unified GUI for all features
- CSV logging for all detections/events

## Features
- **age_gender.py**: Detects age and gender for all faces in images or video (multi-person, robust with MediaPipe).
- **count.py**: Counts unique people using face recognition or body detection.
- **motion.py**: Real-time pose-based motion analysis, gesture and head direction detection, and activity classification (standing, sitting, jumping, walking).
- **thieves.py**: Face recognition for security, with add/remove/list/monitor features.
- **main_interface.py**: Tkinter GUI to launch and control all features.
- **Logger modules**: Each feature logs to its own CSV file.

## Setup

### 1. Clone the repository
```
git clone <your-repo-url>
cd Footfall
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Download required model files
For age/gender and face detection, run:
```
python age_gender.py --download
```
This will download all necessary Caffe models and prototxt files.

For ONNX-based age/gender (optional, for higher accuracy):
- Download the ONNX model from [here](https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/weights.28-3.73.onnx) and place it in the project directory as `age_gender.onnx`.

## Usage

### Main Interface (Recommended)
```
python main_interface.py
```
- Use the GUI to launch any feature.
- For count.py and thieves.py, the interface will prompt for options.

### Individual Scripts
- **Age/Gender Detection:**
  - Webcam: `python age_gender.py --mode webcam`
  - Video: `python age_gender.py --mode video --input path/to/video.mp4`
  - Image: `python age_gender.py --mode image --input path/to/image.jpg`
  - Tracking (multi-person, persistent IDs): `python age_gender.py --mode track`
- **Unique Counting:**
  - `python count.py`
- **Motion Analysis:**
  - `python motion.py`
- **Face Recognition Security:**
  - `python thieves.py --mode monitor`
  - See `python thieves.py --help` for all options.

## Output
- All detections and events are logged to CSV files:
  - `age_gender_log.csv`, `unique_count_log.csv`, `motion_log.csv`, `face_recognition_log.csv`
- Screenshots and result images are saved in the project directory.

## Notes
- For best results, use a well-lit environment and a good webcam.
- All code is modular and can be extended for new features (emotion, mask detection, etc.).

## License
MIT License (or specify your own) 