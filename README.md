# Shadow Coach - C21473436 Mateusz Matijuk TU858/4

System that analyzes shadowboxing videos to detect punches and provide performance metrics using computer vision and machine learning.

---

## Quick Start Guide

### 1. Install Dependencies

**Activate virtual environment and install packages:**

**Windows:**
```bash
python -m venv patk/to/venv
env\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python -m venv patk/to/venv
source env/bin/activate
pip install -r requirements.txt
```

---

### 2. Start the API Server

```bash
python api/app.py
```

Server runs on `http://localhost:5000`

You should see:
```
============================================================
Shadow Coach API Server
============================================================
Model: models\jab_classifier.joblib
Features: 20
============================================================
 * Running on http://0.0.0.0:5000
```

---

### 3. Start the Flutter App

**Navigate to Flutter app directory:**
```bash
cd shadow_coach_app
```

**Run the app:**

**For Web:**
```bash
flutter run -d chrome
```

**For Android/iOS:**
```bash
flutter run
```

The app will connect to the API server at `http://localhost:5000`

---

### 4. Test the System

**Option A: Use the Flutter app**
- Record or upload a shadowboxing video
- View analysis results on the app

**Option B: Test API directly with provided test videos**
```bash
python api/test_api.py
```

**Option C: Use curl**
```bash
curl -X POST http://localhost:5000/analyze -F "video=@data/test/videos/upper-body.mov"
```

---

## Project Contents

```
ShadowCoach/
│
├── api/                          # Flask REST API server
│   ├── app.py                   # Main API server (start with: python api/app.py)
│   └── test_api.py              # Test script for API endpoints
│
├── src/                          # Core Python modules
│   ├── config.py                # Configuration constants
│   ├── video_utils.py           # Video frame extraction
│   ├── pose_extraction.py       # MediaPipe Pose landmark detection
│   ├── feature_engineering.py   # Biomechanical feature computation
│   ├── inference.py             # Punch detection and metrics calculation
│   └── train_jab_model.py       # Model training script
│
├── data/                         # Dataset
│   ├── raw/                     # Training videos (jab/ and non_jab/)
│   ├── processed/               # Processed landmarks and features
│   └── test/videos/             # 5 test videos for evaluation
│
├── models/                       # Trained ML models
│   ├── jab_classifier.joblib    # Random Forest classifier
│   └── feature_names.txt        # Feature names for inference
│
├── shadow_coach_app/             # Flutter mobile/web app
│   └── (Flutter project files)  # Start with: flutter run -d chrome
│
├── notebooks/                    # Jupyter notebooks for data exploration
│   └── explore_features.ipynb   # Feature visualization and analysis
│
├── env/                          # Python virtual environment
│
└── requirements.txt              # Python dependencies
```

---

## Technologies Used

**Backend:**
- Python 3.11
- Flask (REST API)
- MediaPipe (Pose detection)
- OpenCV (Video processing)
- scikit-learn (Random Forest classifier)

**Frontend:**
- Flutter/Dart 
- Hive (Local storage)

**Model:**
- Random Forest

---
