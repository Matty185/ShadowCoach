"""
MediaPipe Pose landmark extraction from videos.
Extracts pose landmarks and saves them to CSV files.
"""
import csv
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

try:
    from src.video_utils import iter_video_frames
    from src.config import (
        TARGET_FPS,
        POSE_MODEL_COMPLEXITY,
        POSE_MIN_DETECTION_CONFIDENCE,
        POSE_MIN_TRACKING_CONFIDENCE,
        POSE_STATIC_IMAGE_MODE,
        POSE_ENABLE_SEGMENTATION,
        LANDMARKS_DIR,
    )
except ImportError:
    from video_utils import iter_video_frames
    from config import (
        TARGET_FPS,
        POSE_MODEL_COMPLEXITY,
        POSE_MIN_DETECTION_CONFIDENCE,
        POSE_MIN_TRACKING_CONFIDENCE,
        POSE_STATIC_IMAGE_MODE,
        POSE_ENABLE_SEGMENTATION,
        LANDMARKS_DIR,
    )


# MediaPipe Pose landmark names (33 landmarks total)
POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def extract_landmarks_from_video(video_path: str, output_csv_path: str) -> None:
    """
    Runs MediaPipe Pose on the video, saves per-frame landmarks to a CSV.
    Each row = one frame.
    Columns: frame_idx, timestamp_sec, and for each landmark:
             {name}_x, {name}_y, {name}_z, {name}_visibility

    Args:
        video_path: Path to input video file
        output_csv_path: Path where CSV will be saved

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be processed
    """
    video_path = Path(video_path)
    output_csv_path = Path(output_csv_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=POSE_STATIC_IMAGE_MODE,
        model_complexity=POSE_MODEL_COMPLEXITY,
        enable_segmentation=POSE_ENABLE_SEGMENTATION,
        min_detection_confidence=POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=POSE_MIN_TRACKING_CONFIDENCE,
    )
    
    # Prepare CSV columns
    columns = ["frame_idx", "timestamp_sec"]
    for name in POSE_LANDMARK_NAMES:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_visibility"])
    
    rows = []
    
    try:
        print(f"Processing video: {video_path.name}")
        frame_count = 0
        
        for frame_idx, timestamp_sec, frame_bgr in iter_video_frames(video_path, TARGET_FPS):
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = pose.process(frame_rgb)
            
            # Initialize row with frame info
            row = [frame_idx, timestamp_sec]
            
            # Extract landmarks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for landmark in landmarks:
                    row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            else:
                # No pose detected - fill with NaNs
                row.extend([np.nan] * (len(POSE_LANDMARK_NAMES) * 4))
            
            rows.append(row)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames...")
        
        print(f"  Total frames processed: {frame_count}")
    
    finally:
        pose.close()
    
    # Create DataFrame and save to CSV
    if not rows:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    df = pd.DataFrame(rows, columns=columns)
    
    # Ensure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv_path, index=False)
    print(f"Saved landmarks to: {output_csv_path}")


def process_video(video_path: str, output_dir: str, label: str = "video", verbose: bool = True) -> Optional[str]:
    """
    Process a single video and extract pose landmarks.
    Convenience function for API usage.

    Args:
        video_path: Path to video file
        output_dir: Directory to save landmark CSV
        label: Label prefix for output file
        verbose: Whether to print progress messages

    Returns:
        Path to the generated landmark CSV file, or None if failed
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{label}_{video_path.stem}_landmarks.csv"
    output_path = output_dir / output_filename

    try:
        # Temporarily suppress print if not verbose
        if not verbose:
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        extract_landmarks_from_video(str(video_path), str(output_path))

        if not verbose:
            sys.stdout = old_stdout

        return str(output_path)
    except Exception as e:
        if not verbose:
            sys.stdout = old_stdout
        print(f"Error processing {video_path.name}: {e}")
        return None


def batch_extract_landmarks(input_dir: str, label: str) -> None:
    """
    For all videos in input_dir, run extract_landmarks_from_video and save to
    LANDMARKS_DIR as <label>_<video_stem>_landmarks.csv

    Args:
        input_dir: Directory containing video files
        label: Label for the videos (e.g., "jab" or "non_jab")
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all video files
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"Warning: No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video(s) in {input_dir}")
    print(f"Label: {label}")
    print("-" * 50)
    
    for video_path in video_files:
        video_stem = video_path.stem
        output_filename = f"{label}_{video_stem}_landmarks.csv"
        output_path = LANDMARKS_DIR / output_filename
        
        try:
            extract_landmarks_from_video(str(video_path), str(output_path))
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            continue
    
    print("-" * 50)
    print(f"Batch extraction complete. Landmarks saved to: {LANDMARKS_DIR}")


if __name__ == "__main__":
    """
    CLI for landmark extraction.
    Usage:
        python -m src.pose_extraction --input_dir data/raw/jab --label jab
        python -m src.pose_extraction --input_dir data/raw/non_jab --label non_jab
    """
    import argparse
    import sys
    import cv2
    
    parser = argparse.ArgumentParser(description="Extract pose landmarks from videos")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing video files"
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label for the videos (e.g., 'jab' or 'non_jab')"
    )
    
    args = parser.parse_args()
    
    try:
        batch_extract_landmarks(args.input_dir, args.label)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

