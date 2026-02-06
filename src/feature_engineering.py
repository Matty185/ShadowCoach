"""
Feature engineering from pose landmarks.
Converts landmark data into ML-ready features.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import math

from src.config import (
    LANDMARKS_DIR,
    FEATURES_DIR,
    FEATURES_DATASET_CSV,
    WINDOW_SIZE_FRAMES,
    WINDOW_STEP_FRAMES,
    LANDMARK_LEFT_SHOULDER,
    LANDMARK_RIGHT_SHOULDER,
    LANDMARK_LEFT_ELBOW,
    LANDMARK_RIGHT_ELBOW,
    LANDMARK_LEFT_WRIST,
    LANDMARK_RIGHT_WRIST,
    LANDMARK_LEFT_HIP,
    LANDMARK_RIGHT_HIP,
)


def compute_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """
    Compute the angle at p2 formed by points p1-p2-p3.
    
    Args:
        p1: First point (x, y)
        p2: Middle point (x, y) - vertex of the angle
        p3: Third point (x, y)
    
    Returns:
        Angle in degrees
    """
    # Vector from p2 to p1
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    # Vector from p2 to p3
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Compute angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def compute_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_per_frame_features(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-frame features from landmarks DataFrame.
    
    Args:
        landmarks_df: DataFrame with columns frame_idx, timestamp_sec, and landmark coordinates
    
    Returns:
        DataFrame with per-frame features
    """
    features_list = []
    
    for idx, row in landmarks_df.iterrows():
        frame_features = {
            "frame_idx": row["frame_idx"],
            "timestamp_sec": row["timestamp_sec"],
        }
        
        # Get landmark coordinates (assuming MediaPipe format)
        landmark_names = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
        ]
        
        landmarks_dict = {}
        for name in landmark_names:
            x = row.get(f"{name}_x", np.nan)
            y = row.get(f"{name}_y", np.nan)
            visibility = row.get(f"{name}_visibility", 0.0)
            landmarks_dict[name] = (x, y, visibility)
        
        # Compute features for left arm (assuming jab is typically with left/lead hand)
        left_shoulder = landmarks_dict["left_shoulder"]
        left_elbow = landmarks_dict["left_elbow"]
        left_wrist = landmarks_dict["left_wrist"]
        right_shoulder = landmarks_dict["right_shoulder"]
        left_hip = landmarks_dict["left_hip"]
        
        # Left elbow angle (shoulder-elbow-wrist)
        if not (np.isnan(left_shoulder[0]) or np.isnan(left_elbow[0]) or np.isnan(left_wrist[0])):
            if left_shoulder[2] > 0.5 and left_elbow[2] > 0.5 and left_wrist[2] > 0.5:
                angle = compute_angle(
                    (left_shoulder[0], left_shoulder[1]),
                    (left_elbow[0], left_elbow[1]),
                    (left_wrist[0], left_wrist[1])
                )
                frame_features["left_elbow_angle"] = angle
            else:
                frame_features["left_elbow_angle"] = np.nan
        else:
            frame_features["left_elbow_angle"] = np.nan
        
        # Left wrist to left shoulder distance
        if not (np.isnan(left_wrist[0]) or np.isnan(left_shoulder[0])):
            if left_wrist[2] > 0.5 and left_shoulder[2] > 0.5:
                dist = compute_distance(
                    (left_wrist[0], left_wrist[1]),
                    (left_shoulder[0], left_shoulder[1])
                )
                frame_features["left_wrist_shoulder_dist"] = dist
            else:
                frame_features["left_wrist_shoulder_dist"] = np.nan
        else:
            frame_features["left_wrist_shoulder_dist"] = np.nan
        
        # Left wrist to left hip distance
        if not (np.isnan(left_wrist[0]) or np.isnan(left_hip[0])):
            if left_wrist[2] > 0.5 and left_hip[2] > 0.5:
                dist = compute_distance(
                    (left_wrist[0], left_wrist[1]),
                    (left_hip[0], left_hip[1])
                )
                frame_features["left_wrist_hip_dist"] = dist
            else:
                frame_features["left_wrist_hip_dist"] = np.nan
        else:
            frame_features["left_wrist_hip_dist"] = np.nan
        
        # Right arm features (for comparison)
        right_elbow = landmarks_dict["right_elbow"]
        right_wrist = landmarks_dict["right_wrist"]
        
        if not (np.isnan(right_shoulder[0]) or np.isnan(right_elbow[0]) or np.isnan(right_wrist[0])):
            if right_shoulder[2] > 0.5 and right_elbow[2] > 0.5 and right_wrist[2] > 0.5:
                angle = compute_angle(
                    (right_shoulder[0], right_shoulder[1]),
                    (right_elbow[0], right_elbow[1]),
                    (right_wrist[0], right_wrist[1])
                )
                frame_features["right_elbow_angle"] = angle
            else:
                frame_features["right_elbow_angle"] = np.nan
        else:
            frame_features["right_elbow_angle"] = np.nan
        
        features_list.append(frame_features)
    
    return pd.DataFrame(features_list)


def compute_velocity_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute frame-to-frame velocity features.
    
    Args:
        features_df: DataFrame with per-frame features including positions
    
    Returns:
        DataFrame with added velocity features
    """
    df = features_df.copy()
    
    # Sort by frame_idx to ensure correct order
    df = df.sort_values("frame_idx").reset_index(drop=True)
    
    # Compute wrist velocity (magnitude of change in position)
    if "left_wrist_shoulder_dist" in df.columns:
        # Use change in distance as a proxy for velocity
        df["left_wrist_velocity"] = df["left_wrist_shoulder_dist"].diff().abs()
        df["left_wrist_velocity"].fillna(0, inplace=True)
    
    return df


def compute_features_from_landmarks(
    landmarks_df: pd.DataFrame,
    window_size: int,
    step: int
) -> pd.DataFrame:
    """
    Returns a DataFrame where each row is a time window, with aggregated features:
    - mean / max wrist velocity in window
    - mean elbow angle
    - max wrist-shoulder distance
    - etc.

    Args:
        landmarks_df: DataFrame with landmark data
        window_size: Number of frames in each sliding window
        step: Step size between windows

    Returns:
        DataFrame with windowed features
    """
    # Compute per-frame features
    per_frame_features = compute_per_frame_features(landmarks_df)
    per_frame_features = compute_velocity_features(per_frame_features)
    
    # Apply sliding window
    window_features = []
    
    for start_idx in range(0, len(per_frame_features) - window_size + 1, step):
        window_df = per_frame_features.iloc[start_idx:start_idx + window_size]
        
        window_feat = {}
        
        # Aggregate features over the window
        for col in ["left_elbow_angle", "right_elbow_angle", 
                   "left_wrist_shoulder_dist", "left_wrist_hip_dist",
                   "left_wrist_velocity"]:
            if col in window_df.columns:
                window_feat[f"{col}_mean"] = window_df[col].mean()
                window_feat[f"{col}_max"] = window_df[col].max()
                window_feat[f"{col}_min"] = window_df[col].min()
                window_feat[f"{col}_std"] = window_df[col].std()
        
        # Add window metadata
        window_feat["window_start_frame"] = start_idx
        window_feat["window_end_frame"] = start_idx + window_size - 1
        window_feat["window_start_time"] = window_df["timestamp_sec"].min()
        window_feat["window_end_time"] = window_df["timestamp_sec"].max()
        
        window_features.append(window_feat)
    
    if not window_features:
        return pd.DataFrame()
    
    return pd.DataFrame(window_features)


def build_feature_dataset(landmarks_dir: str, output_csv_path: str) -> None:
    """
    Iterates over all landmarks CSVs in landmarks_dir, computes features,
    assigns labels from filename (contains 'jab' or 'non_jab'), and writes a combined dataset CSV.

    Args:
        landmarks_dir: Directory containing landmark CSV files
        output_csv_path: Path where combined feature dataset will be saved
    """
    landmarks_dir = Path(landmarks_dir)
    output_csv_path = Path(output_csv_path)
    
    if not landmarks_dir.exists():
        raise FileNotFoundError(f"Landmarks directory not found: {landmarks_dir}")
    
    # Find all landmark CSV files
    landmark_files = list(landmarks_dir.glob("*_landmarks.csv"))
    
    if not landmark_files:
        raise ValueError(f"No landmark CSV files found in {landmarks_dir}")
    
    print(f"Found {len(landmark_files)} landmark file(s)")
    print("-" * 50)
    
    all_features = []
    
    for landmark_file in landmark_files:
        print(f"Processing: {landmark_file.name}")
        
        # Determine label from filename
        filename_lower = landmark_file.stem.lower()
        if "jab" in filename_lower and "non_jab" not in filename_lower:
            label = 1
        elif "non_jab" in filename_lower:
            label = 0
        else:
            print(f"  Warning: Could not determine label from filename, skipping")
            continue
        
        try:
            # Load landmarks
            landmarks_df = pd.read_csv(landmark_file)
            
            if len(landmarks_df) == 0:
                print(f"  Warning: Empty landmarks file, skipping")
                continue
            
            # Compute windowed features
            windowed_features = compute_features_from_landmarks(
                landmarks_df,
                WINDOW_SIZE_FRAMES,
                WINDOW_STEP_FRAMES
            )
            
            if len(windowed_features) == 0:
                print(f"  Warning: No windows extracted, skipping")
                continue
            
            # Add label and source file
            windowed_features["label"] = label
            windowed_features["source_file"] = landmark_file.name
            
            all_features.append(windowed_features)
            print(f"  Extracted {len(windowed_features)} windows")
        
        except Exception as e:
            print(f"  Error processing {landmark_file.name}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No features extracted from any landmark files")
    
    # Combine all features
    combined_df = pd.concat(all_features, ignore_index=True)
    
    # Ensure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save combined dataset
    combined_df.to_csv(output_csv_path, index=False)
    
    print("-" * 50)
    print(f"Feature dataset saved to: {output_csv_path}")
    print(f"Total windows: {len(combined_df)}")
    print(f"Jab windows (label=1): {len(combined_df[combined_df['label'] == 1])}")
    print(f"Non-jab windows (label=0): {len(combined_df[combined_df['label'] == 0])}")


if __name__ == "__main__":
    """
    CLI for feature engineering.
    Usage: python -m src.feature_engineering
    """
    import sys
    
    try:
        build_feature_dataset(str(LANDMARKS_DIR), str(FEATURES_DATASET_CSV))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

