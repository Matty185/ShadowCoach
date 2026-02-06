"""
Flask REST API for Shadow Coach jab detection.
Provides endpoints for video analysis and model inference.
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import MODELS_DIR, LANDMARKS_DIR
from pose_extraction import process_video
from inference import (
    analyze_video, 
    merge_overlapping_jabs, 
    merge_overlapping_punches,
    calculate_comprehensive_metrics,
    compute_per_frame_features,
    detect_generic_punches
)
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max file size

# Enable CORS for Flutter web app
CORS(app)

# Allowed video formats
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# Load model at startup
model_path = MODELS_DIR / "jab_classifier.joblib"
feature_names_path = MODELS_DIR / "feature_names.txt"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

model = joblib.load(model_path)

with open(feature_names_path, 'r') as f:
    feature_names = [line.strip() for line in f if line.strip()]

print(f"[OK] Model loaded successfully ({len(feature_names)} features)")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Shadow Coach API',
        'model_loaded': True,
        'features': len(feature_names)
    }), 200


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze uploaded video for jab detection.

    Expected: multipart/form-data with 'video' file
    Returns: JSON with jab detection results
    """
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']

        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(video_file.filename):
            return jsonify({
                'error': f'Invalid file format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Save uploaded file to temporary location
        filename = secure_filename(video_file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_video_path = Path(temp_dir) / filename
        video_file.save(str(temp_video_path))

        print(f"\n[API] Processing video: {filename}")
        print(f"[API] File size: {temp_video_path.stat().st_size / (1024*1024):.2f} MB")

        # Step 1: Extract pose landmarks
        print("[API] Step 1/3: Extracting pose landmarks...")
        temp_landmarks_dir = Path(temp_dir) / "landmarks"
        temp_landmarks_dir.mkdir(exist_ok=True)

        landmark_file = process_video(
            video_path=str(temp_video_path),
            output_dir=str(temp_landmarks_dir),
            label="api",
            verbose=False
        )

        if not landmark_file or not Path(landmark_file).exists():
            return jsonify({'error': 'Failed to extract pose landmarks'}), 500

        print(f"[API] [OK] Landmarks extracted: {landmark_file}")

        # Step 2: Load landmark data and compute features
        print("[API] Step 2/5: Loading landmark data and computing features...")
        landmark_df = pd.read_csv(landmark_file)
        feature_df = compute_per_frame_features(landmark_df)

        # Step 3: Run jab detection (model-based)
        print("[API] Step 3/5: Running jab detection model...")
        jab_windows = analyze_video(Path(landmark_file), model, feature_names, fps=15)
        merged_jabs = merge_overlapping_jabs(jab_windows) if len(jab_windows) > 0 else []
        
        # Add punch_type label to jabs
        for jab in merged_jabs:
            jab['punch_type'] = 'jab'
            # Try to determine hand from features if not already set
            if 'hand' not in jab:
                jab['hand'] = 'left'  # Default for current model (only left jab detection)

        # Step 4: Run generic punch detection (motion-based)
        print("[API] Step 4/5: Running motion-based punch detection...")
        generic_punches = detect_generic_punches(feature_df, landmark_df, fps=15)
        
        # Add punch_type label to generic punches
        for punch in generic_punches:
            punch['punch_type'] = 'punch'

        # Step 5: Merge all punches (prefer jab label when overlapping)
        print("[API] Step 5/5: Merging detections and calculating metrics...")
        all_punches = merged_jabs + generic_punches
        merged_punches = merge_overlapping_punches(all_punches, prefer_jab=True) if len(all_punches) > 0 else []

        # Calculate comprehensive metrics using all punches
        comprehensive_metrics = calculate_comprehensive_metrics(merged_punches, landmark_df, fps=15)

        # Build punch events array with punch types
        punch_events = []
        for i, punch in enumerate(merged_punches, 1):
            # Calculate speed: for jabs use confidence scaling, for generic punches use velocity-based
            if punch.get('punch_type') == 'jab':
                speed = punch['confidence'] * 5  # Scale confidence to speed metric (0-5 m/s range)
            else:
                # For generic punches, use confidence (which is based on velocity) scaled similarly
                speed = punch['confidence'] * 5
            
            punch_events.append({
                'index': i,
                'speed': speed,
                'hand': punch.get('hand', 'left'),  # Use detected hand or default to left
                'start': punch['start_time'],
                'end': punch['end_time'],
                'duration': punch['duration'],
                'confidence': punch['confidence'],
                'punch_type': punch.get('punch_type', 'punch')  # 'jab' or 'punch'
            })

        # Calculate hand distribution from all punches
        left_count = sum(1 for p in merged_punches if p.get('hand') == 'left')
        right_count = sum(1 for p in merged_punches if p.get('hand') == 'right')
        hand_distribution = {
            'left': left_count,
            'right': right_count
        }

        # Build response with comprehensive metrics
        response = {
            # Basic metrics
            'total_punches': comprehensive_metrics['total_punches'],
            'average_speed': comprehensive_metrics['average_speed'],
            'punches_per_minute': comprehensive_metrics['punches_per_minute'],
            'punch_events': punch_events,

            # Session info
            'session_duration': comprehensive_metrics['session_duration'],

            # Performance metrics
            'performance': {
                'max_speed': comprehensive_metrics['max_speed'],
                'min_speed': comprehensive_metrics['min_speed'],
                'speed_variance': comprehensive_metrics['speed_variance'],
                'average_punch_duration': comprehensive_metrics['average_punch_duration'],
                'fastest_punch': comprehensive_metrics.get('fastest_punch'),
                'slowest_punch': comprehensive_metrics.get('slowest_punch')
            },

            # Temporal/Activity metrics
            'activity': {
                'active_time': comprehensive_metrics['active_time'],
                'rest_time': comprehensive_metrics['rest_time'],
                'activity_percentage': comprehensive_metrics['activity_percentage'],
                'average_rest_period': comprehensive_metrics['average_rest_period'],
                'max_rest_period': comprehensive_metrics['max_rest_period'],
                'min_rest_period': comprehensive_metrics.get('min_rest_period', 0)
            },

            # Intensity metrics
            'intensity': {
                'score': comprehensive_metrics['punch_intensity_score'],
                'punches_by_interval': comprehensive_metrics['punches_by_interval'],
                'peak_interval': comprehensive_metrics.get('peak_interval')
            },

            # Fatigue indicators
            'fatigue': {
                'indicator': comprehensive_metrics['fatigue_indicator'],
                'speed_trend': comprehensive_metrics['speed_trend']
            },

            # Graphs (for visualization)
            'graphs': {
                'speed_over_time': [jab['confidence'] * 5 for jab in merged_jabs],
                'hand_distribution': hand_distribution,
                'punches_by_interval': comprehensive_metrics['punches_by_interval']
            }
        }

        # Cleanup temporary files
        try:
            temp_video_path.unlink()
            Path(landmark_file).unlink()
            os.rmdir(str(temp_landmarks_dir))
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"[API] Warning: Failed to cleanup temp files: {e}")

        print(f"[API] [OK] Analysis complete: {comprehensive_metrics['total_punches']} jabs detected")

        return jsonify(response), 200

    except Exception as e:
        print(f"[API] [ERROR] Error during analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'Shadow Coach API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/analyze': 'POST - Analyze video for jab detection (multipart/form-data with "video" file)'
        }
    }), 200


if __name__ == '__main__':
    print("=" * 60)
    print("Shadow Coach API Server")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Features: {len(feature_names)}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
