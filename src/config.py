"""
Configuration file for the jab detection pipeline.
Centralizes paths and constants used throughout the project.
"""
import os
from pathlib import Path

# Base directory (project root)
PROJECT_ROOT = Path(__file__).parent.parent

# Raw video directories
RAW_JAB_DIR = PROJECT_ROOT / "data" / "raw" / "jab"
RAW_NONJAB_DIR = PROJECT_ROOT / "data" / "raw" / "non_jab"

# Processed data directories
LANDMARKS_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Video processing settings
TARGET_FPS = 15  # Target frames per second for frame extraction

# MediaPipe Pose settings
POSE_MODEL_COMPLEXITY = 2 
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_STATIC_IMAGE_MODE = False
POSE_ENABLE_SEGMENTATION = False

# MediaPipe Pose landmark indices for jab detection
# Key landmarks: shoulders, elbows, wrists, hips
LANDMARK_LEFT_SHOULDER = 11
LANDMARK_RIGHT_SHOULDER = 12
LANDMARK_LEFT_ELBOW = 13
LANDMARK_RIGHT_ELBOW = 14
LANDMARK_LEFT_WRIST = 15
LANDMARK_RIGHT_WRIST = 16
LANDMARK_LEFT_HIP = 23
LANDMARK_RIGHT_HIP = 24

# Feature extraction settings
WINDOW_SIZE_FRAMES = 15  # Number of frames in each sliding window
WINDOW_STEP_FRAMES = 5  # Step size between windows 

# Model training settings
TEST_SIZE = 0.2  # Fraction of data to use for testing
RANDOM_STATE = 42  # Random seed for reproducibility

# Output filenames
FEATURES_DATASET_CSV = FEATURES_DIR / "jab_nonjab_dataset.csv"
MODEL_FILENAME = MODELS_DIR / "jab_classifier.joblib"

# Ensure directories exist
for directory in [RAW_JAB_DIR, RAW_NONJAB_DIR, LANDMARKS_DIR, FEATURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)