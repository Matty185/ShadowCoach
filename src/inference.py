"""
Inference script for jab detection on new videos.
Processes test videos and outputs detected jab events.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    WINDOW_SIZE_FRAMES,
    WINDOW_STEP_FRAMES,
    PUNCH_REGION_MARGIN_FRAMES,
    WINDOW_OVERLAP_THRESHOLD,
    LANDMARKS_DIR,
    MODELS_DIR
)


def compute_per_frame_features(df):
    """Compute biomechanical features for each frame."""
    # Elbow angles
    def compute_angle(a, b, c):
        """Compute angle at point b given three points a, b, c."""
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    features = []
    for idx, row in df.iterrows():
        # Left arm
        left_shoulder = np.array([row['left_shoulder_x'], row['left_shoulder_y']])
        left_elbow = np.array([row['left_elbow_x'], row['left_elbow_y']])
        left_wrist = np.array([row['left_wrist_x'], row['left_wrist_y']])

        # Right arm
        right_shoulder = np.array([row['right_shoulder_x'], row['right_shoulder_y']])
        right_elbow = np.array([row['right_elbow_x'], row['right_elbow_y']])
        right_wrist = np.array([row['right_wrist_x'], row['right_wrist_y']])

        # Hips
        left_hip = np.array([row['left_hip_x'], row['left_hip_y']])

        # Compute angles
        left_elbow_angle = compute_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = compute_angle(right_shoulder, right_elbow, right_wrist)

        # Compute distances
        left_wrist_shoulder_dist = np.linalg.norm(left_wrist - left_shoulder)
        left_wrist_hip_dist = np.linalg.norm(left_wrist - left_hip)

        features.append({
            'frame': row['frame_idx'],
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle,
            'left_wrist_shoulder_dist': left_wrist_shoulder_dist,
            'left_wrist_hip_dist': left_wrist_hip_dist,
            'left_wrist_x': left_wrist[0],
            'left_wrist_y': left_wrist[1]
        })

    feature_df = pd.DataFrame(features)

    # Compute velocities (frame-to-frame change) for both hands
    feature_df['left_wrist_velocity'] = np.sqrt(
        feature_df['left_wrist_x'].diff()**2 +
        feature_df['left_wrist_y'].diff()**2
    ).fillna(0)
    
    # Compute right wrist velocity (need to extract right wrist positions)
    right_wrist_x = []
    right_wrist_y = []
    for idx, row in df.iterrows():
        right_wrist = np.array([row['right_wrist_x'], row['right_wrist_y']])
        right_wrist_x.append(right_wrist[0])
        right_wrist_y.append(right_wrist[1])
    
    feature_df['right_wrist_x'] = right_wrist_x
    feature_df['right_wrist_y'] = right_wrist_y
    feature_df['right_wrist_velocity'] = np.sqrt(
        feature_df['right_wrist_x'].diff()**2 +
        feature_df['right_wrist_y'].diff()**2
    ).fillna(0)

    return feature_df


def create_windows(feature_df, window_size=WINDOW_SIZE_FRAMES, step=WINDOW_STEP_FRAMES):
    """Create sliding windows from features."""
    windows = []
    num_frames = len(feature_df)

    for start_idx in range(0, num_frames - window_size + 1, step):
        end_idx = start_idx + window_size
        window = feature_df.iloc[start_idx:end_idx]

        # Aggregate features
        window_features = {
            'start_frame': window['frame'].iloc[0],
            'end_frame': window['frame'].iloc[-1],
            'left_elbow_angle_mean': window['left_elbow_angle'].mean(),
            'left_elbow_angle_max': window['left_elbow_angle'].max(),
            'left_elbow_angle_min': window['left_elbow_angle'].min(),
            'left_elbow_angle_std': window['left_elbow_angle'].std(),
            'right_elbow_angle_mean': window['right_elbow_angle'].mean(),
            'right_elbow_angle_max': window['right_elbow_angle'].max(),
            'right_elbow_angle_min': window['right_elbow_angle'].min(),
            'right_elbow_angle_std': window['right_elbow_angle'].std(),
            'left_wrist_shoulder_dist_mean': window['left_wrist_shoulder_dist'].mean(),
            'left_wrist_shoulder_dist_max': window['left_wrist_shoulder_dist'].max(),
            'left_wrist_shoulder_dist_min': window['left_wrist_shoulder_dist'].min(),
            'left_wrist_shoulder_dist_std': window['left_wrist_shoulder_dist'].std(),
            'left_wrist_hip_dist_mean': window['left_wrist_hip_dist'].mean(),
            'left_wrist_hip_dist_max': window['left_wrist_hip_dist'].max(),
            'left_wrist_hip_dist_min': window['left_wrist_hip_dist'].min(),
            'left_wrist_hip_dist_std': window['left_wrist_hip_dist'].std(),
            'left_wrist_velocity_mean': window['left_wrist_velocity'].mean(),
            'left_wrist_velocity_max': window['left_wrist_velocity'].max(),
            'left_wrist_velocity_min': window['left_wrist_velocity'].min(),
            'left_wrist_velocity_std': window['left_wrist_velocity'].std(),
        }
        windows.append(window_features)

    return pd.DataFrame(windows)


def create_windows_for_punch_regions(feature_df, punch_regions,
                                      window_size=WINDOW_SIZE_FRAMES,
                                      step=WINDOW_STEP_FRAMES,
                                      margin_frames=PUNCH_REGION_MARGIN_FRAMES):
    """
    Create sliding windows only for detected punch regions (optimized approach).

    This function maps punch events (frame ranges) to sliding windows,
    extracting features only for windows that overlap with or are near detected punches.
    This significantly reduces the number of ML inferences needed.

    Args:
        feature_df: Per-frame features DataFrame with 'frame' column
        punch_regions: List of punch dictionaries with 'start_frame' and 'end_frame'
        window_size: Number of frames per window (default: 15)
        step: Step size between windows (default: 5)
        margin_frames: Extra frames to include before/after each punch (default: 10)

    Returns:
        DataFrame of aggregated window features for detected punch regions only
        (same format as create_windows())

    Window-to-Punch Mapping Logic:
        - For each punch [start_frame, end_frame]:
          - Expand region by margin: [start_frame - margin, end_frame + margin]
          - Find all windows that overlap with this expanded region
          - A window [w_start, w_start + window_size] overlaps if:
            w_start + window_size >= region_start AND w_start <= region_end

    Performance:
        - Legacy approach: Process ALL frames (~87 windows for 30s video)
        - Optimized approach: Process only punch regions (~15-20 windows)
        - Reduction: 70-90% fewer ML inference calls
    """
    if len(punch_regions) == 0:
        return pd.DataFrame()  # No punches, return empty DataFrame

    if len(feature_df) == 0:
        return pd.DataFrame()

    num_frames = len(feature_df)
    max_frame = feature_df['frame'].max()

    # Track unique windows to avoid duplicates
    window_set = set()

    # For each punch region, find overlapping windows
    for punch in punch_regions:
        # Expand punch region by margin
        region_start = max(0, punch['start_frame'] - margin_frames)
        region_end = min(max_frame, punch['end_frame'] + margin_frames)

        # Find all windows that overlap with this expanded region
        for window_start_frame in range(0, num_frames - window_size + 1, step):
            window_start_idx = window_start_frame
            window_end_frame = feature_df.iloc[window_start_idx]['frame'] + window_size - 1

            # Check if this window overlaps with the punch region
            # Window overlaps if: window_end >= region_start AND window_start <= region_end
            if window_end_frame >= region_start and feature_df.iloc[window_start_idx]['frame'] <= region_end:
                # Add to set (using tuple to track unique windows)
                window_set.add((window_start_idx, window_start_idx + window_size))

    # Now create windowed features for all unique windows
    windows = []
    for start_idx, end_idx in sorted(window_set):
        # Ensure indices are within bounds
        if end_idx > len(feature_df):
            end_idx = len(feature_df)

        window = feature_df.iloc[start_idx:end_idx]

        if len(window) < window_size:
            # Skip incomplete windows
            continue

        # Aggregate features (same as create_windows())
        window_features = {
            'start_frame': window['frame'].iloc[0],
            'end_frame': window['frame'].iloc[-1],
            'left_elbow_angle_mean': window['left_elbow_angle'].mean(),
            'left_elbow_angle_max': window['left_elbow_angle'].max(),
            'left_elbow_angle_min': window['left_elbow_angle'].min(),
            'left_elbow_angle_std': window['left_elbow_angle'].std(),
            'right_elbow_angle_mean': window['right_elbow_angle'].mean(),
            'right_elbow_angle_max': window['right_elbow_angle'].max(),
            'right_elbow_angle_min': window['right_elbow_angle'].min(),
            'right_elbow_angle_std': window['right_elbow_angle'].std(),
            'left_wrist_shoulder_dist_mean': window['left_wrist_shoulder_dist'].mean(),
            'left_wrist_shoulder_dist_max': window['left_wrist_shoulder_dist'].max(),
            'left_wrist_shoulder_dist_min': window['left_wrist_shoulder_dist'].min(),
            'left_wrist_shoulder_dist_std': window['left_wrist_shoulder_dist'].std(),
            'left_wrist_hip_dist_mean': window['left_wrist_hip_dist'].mean(),
            'left_wrist_hip_dist_max': window['left_wrist_hip_dist'].max(),
            'left_wrist_hip_dist_min': window['left_wrist_hip_dist'].min(),
            'left_wrist_hip_dist_std': window['left_wrist_hip_dist'].std(),
            'left_wrist_velocity_mean': window['left_wrist_velocity'].mean(),
            'left_wrist_velocity_max': window['left_wrist_velocity'].max(),
            'left_wrist_velocity_min': window['left_wrist_velocity'].min(),
            'left_wrist_velocity_std': window['left_wrist_velocity'].std(),
        }
        windows.append(window_features)

    return pd.DataFrame(windows)


def analyze_video(landmark_file, model, feature_names, fps=15):
    """Analyze a single video and return detected jab events."""
    print(f"\nAnalyzing: {landmark_file.name}")

    # Load landmarks
    df = pd.read_csv(landmark_file)

    # Compute features
    feature_df = compute_per_frame_features(df)

    # Create windows
    window_df = create_windows(feature_df)

    if len(window_df) == 0:
        print("  [WARNING] Video too short - no windows created")
        return []

    # Extract feature columns in correct order
    X = window_df[feature_names]

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Get jab windows (prediction = 1)
    jab_windows = window_df[predictions == 1].copy()
    jab_windows['confidence'] = probabilities[predictions == 1, 1]

    # Convert frames to timestamps
    jab_windows['start_time'] = jab_windows['start_frame'] / fps
    jab_windows['end_time'] = jab_windows['end_frame'] / fps
    jab_windows['duration'] = jab_windows['end_time'] - jab_windows['start_time']

    print(f"  Total windows analyzed: {len(window_df)}")
    print(f"  Jab windows detected: {len(jab_windows)}")

    return jab_windows[['start_frame', 'end_frame', 'start_time', 'end_time', 'duration', 'confidence']]


def analyze_video_optimized(feature_df, landmark_df, model, feature_names, fps=15):
    """
    Optimized video analysis: Run generic detection first, then ML classification
    on detected punch regions only.

    This is the main optimization that reduces ML inference calls by 70-90%.

    PIPELINE:
    1. Generic Punch Detection (motion-based) → Identifies punch candidates
    2. Targeted Window Extraction → Creates windows only around detected punches
    3. ML Classification → Runs model only on targeted windows
    4. Result Mapping → Maps ML predictions back to punch events

    Args:
        feature_df: Per-frame features DataFrame
        landmark_df: Original landmark DataFrame
        model: Trained ML classifier (RandomForest)
        feature_names: List of feature names for model input
        fps: Frames per second (default: 15)

    Returns:
        Tuple of (ml_classified_punches, unclassified_punches):
        - ml_classified_punches: Punches classified by ML as jabs (with punch_type='jab')
        - unclassified_punches: Generic punches not classified as jabs (punch_type='punch')

    Performance Example (30s video, 10 punches):
        - Legacy: 87 ML predictions (all windows)
        - Optimized: 40 ML predictions (only punch windows)
        - Reduction: 54% fewer ML calls
    """
    print(f"[OPTIMIZED PIPELINE] Starting optimized analysis...")

    # Step 1: Run generic punch detection first
    print(f"[Step 1/4] Running generic punch detection (motion-based)...")
    generic_punches = detect_generic_punches(feature_df, landmark_df, fps=fps)
    print(f"  - Detected {len(generic_punches)} punch candidate(s)")

    if len(generic_punches) == 0:
        print(f"  - No punches detected, skipping ML classification")
        return [], []  # No punches, no need for ML

    # Step 2: Create targeted windows only for detected punch regions
    print(f"[Step 2/4] Creating targeted windows for punch regions...")
    targeted_windows = create_windows_for_punch_regions(
        feature_df=feature_df,
        punch_regions=generic_punches,
        window_size=WINDOW_SIZE_FRAMES,
        step=WINDOW_STEP_FRAMES,
        margin_frames=PUNCH_REGION_MARGIN_FRAMES
    )

    print(f"  - Created {len(targeted_windows)} targeted window(s)")

    if len(targeted_windows) == 0:
        print(f"  - No valid windows created, returning generic punches as-is")
        # Add punch_type to generic punches
        for punch in generic_punches:
            punch['punch_type'] = 'punch'
        return [], generic_punches

    # Step 3: Run ML classification on targeted windows only
    print(f"[Step 3/4] Running ML classification on targeted windows...")

    # Calculate optimization ratio
    total_frames = len(feature_df)
    total_possible_windows = max(1, (total_frames - WINDOW_SIZE_FRAMES + 1) // WINDOW_STEP_FRAMES)
    optimization_ratio = (1 - len(targeted_windows) / total_possible_windows) * 100 if total_possible_windows > 0 else 0

    print(f"  - Optimization: Processing {len(targeted_windows)}/{total_possible_windows} windows ({optimization_ratio:.1f}% reduction)")

    # Extract features in correct order
    X = targeted_windows[feature_names]

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Filter for jabs (prediction == 1)
    jab_mask = predictions == 1
    jab_windows = targeted_windows[jab_mask].copy()

    if len(jab_windows) > 0:
        jab_windows['confidence'] = probabilities[jab_mask, 1]
        jab_windows['punch_type'] = 'jab'
        jab_windows['hand'] = 'left'  # Current model only detects left jabs

        # Convert frames to timestamps
        jab_windows['start_time'] = jab_windows['start_frame'] / fps
        jab_windows['end_time'] = jab_windows['end_frame'] / fps
        jab_windows['duration'] = jab_windows['end_time'] - jab_windows['start_time']

        print(f"  - ML classified {len(jab_windows)} window(s) as jabs")
    else:
        jab_windows = pd.DataFrame()
        print(f"  - No jabs classified by ML")

    # Step 4: Map ML results back to generic punch events
    print(f"[Step 4/4] Mapping ML results to punch events...")

    ml_classified_punches = []
    unclassified_punches = []

    for punch in generic_punches:
        # Check if any jab window overlaps with this punch
        overlapping_jabs = []

        if len(jab_windows) > 0:
            for _, jab in jab_windows.iterrows():
                if windows_overlap(
                    int(jab['start_frame']), int(jab['end_frame']),
                    punch['start_frame'], punch['end_frame'],
                    threshold=WINDOW_OVERLAP_THRESHOLD
                ):
                    overlapping_jabs.append(jab)

        if overlapping_jabs:
            # This punch was classified as a jab by ML
            # Use the jab window with highest confidence
            best_jab = max(overlapping_jabs, key=lambda x: x['confidence'])

            classified_punch = {
                **punch,  # Keep original timing from generic detection
                'punch_type': 'jab',
                'confidence': max(punch['confidence'], float(best_jab['confidence'])),
                'hand': 'left'  # From ML classification
            }
            ml_classified_punches.append(classified_punch)
        else:
            # Generic punch that wasn't classified as jab
            punch['punch_type'] = 'punch'  # Generic punch (not jab)
            unclassified_punches.append(punch)

    print(f"  - Final results: {len(ml_classified_punches)} jab(s), {len(unclassified_punches)} generic punch(es)")
    print(f"[OPTIMIZED PIPELINE] Analysis complete!")

    return ml_classified_punches, unclassified_punches


def detect_generic_punches(feature_df, landmark_df, fps=15, velocity_threshold=0.08, min_punch_duration=0.1, max_punch_duration=1.0):
    """
    Detect generic punch events using motion-based analysis (wrist velocity).
    This detects any rapid forward arm movements, not just jabs.
    
    Args:
        feature_df: DataFrame with per-frame features including wrist velocities
        landmark_df: Original landmark DataFrame for frame/timestamp mapping
        fps: Frames per second
        velocity_threshold: Minimum wrist velocity to consider a punch (normalized units)
        min_punch_duration: Minimum duration for a valid punch (seconds)
        max_punch_duration: Maximum duration for a valid punch (seconds)
    
    Returns:
        List of dictionaries with punch windows: [{
            'start_frame', 'end_frame', 'start_time', 'end_time', 
            'duration', 'confidence', 'hand'
        }]
    """
    if len(feature_df) == 0:
        return []
    
    punch_events = []
    
    # Detect punches for both left and right hands
    for hand in ['left', 'right']:
        velocity_col = f'{hand}_wrist_velocity'
        
        if velocity_col not in feature_df.columns:
            continue
        
        # Find rows where velocity exceeds threshold
        high_velocity_mask = feature_df[velocity_col] > velocity_threshold
        high_velocity_indices = feature_df[high_velocity_mask].index.tolist()
        
        if not high_velocity_indices:
            continue
        
        # Group consecutive high-velocity frames into punch events
        current_punch_start_idx = None
        current_punch_end_idx = None
        
        for i, df_idx in enumerate(high_velocity_indices):
            if current_punch_start_idx is None:
                # Start new punch
                current_punch_start_idx = df_idx
                current_punch_end_idx = df_idx
            elif df_idx == high_velocity_indices[i-1] + 1:
                # Consecutive frame, extend current punch
                current_punch_end_idx = df_idx
            else:
                # Gap detected, save current punch and start new one
                if current_punch_start_idx is not None:
                    start_frame = feature_df.iloc[current_punch_start_idx]['frame']
                    end_frame = feature_df.iloc[current_punch_end_idx]['frame']
                    start_time = start_frame / fps
                    end_time = end_frame / fps
                    duration = end_time - start_time
                    
                    # Filter by duration
                    if min_punch_duration <= duration <= max_punch_duration:
                        # Calculate average velocity as confidence (normalized to 0-1)
                        punch_window = feature_df.iloc[current_punch_start_idx:current_punch_end_idx+1]
                        avg_velocity = punch_window[velocity_col].mean()
                        # Normalize confidence (velocity_threshold to 2*velocity_threshold maps to 0.5-1.0)
                        confidence = min(1.0, max(0.5, (avg_velocity - velocity_threshold) / velocity_threshold))
                        
                        punch_events.append({
                            'start_frame': int(start_frame),
                            'end_frame': int(end_frame),
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': duration,
                            'confidence': confidence,
                            'hand': hand
                        })
                
                current_punch_start_idx = df_idx
                current_punch_end_idx = df_idx
        
        # Include the last punch
        if current_punch_start_idx is not None:
            start_frame = feature_df.iloc[current_punch_start_idx]['frame']
            end_frame = feature_df.iloc[current_punch_end_idx]['frame']
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            
            if min_punch_duration <= duration <= max_punch_duration:
                punch_window = feature_df.iloc[current_punch_start_idx:current_punch_end_idx+1]
                avg_velocity = punch_window[velocity_col].mean()
                confidence = min(1.0, max(0.5, (avg_velocity - velocity_threshold) / velocity_threshold))
                
                punch_events.append({
                    'start_frame': int(start_frame),
                    'end_frame': int(end_frame),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'confidence': confidence,
                    'hand': hand
                })
    
    # Sort by start_time
    punch_events.sort(key=lambda x: x['start_time'])
    
    return punch_events


def merge_overlapping_punches(punch_list, overlap_threshold=0.5, prefer_jab=True):
    """
    Merge overlapping punch detections.
    If prefer_jab is True, when a jab and generic punch overlap, keep the jab label.
    """
    if len(punch_list) == 0:
        return []

    # Sort by start_time
    punches = sorted(punch_list, key=lambda x: x['start_time'])
    merged = []
    current = punches[0].copy()

    for next_punch in punches[1:]:
        # Check if overlapping
        overlap_start = max(current['start_time'], next_punch['start_time'])
        overlap_end = min(current['end_time'], next_punch['end_time'])
        overlap_duration = max(0, overlap_end - overlap_start)

        min_duration = min(current['duration'], next_punch['duration'])

        if overlap_duration / min_duration >= overlap_threshold:
            # Merge - extend current to include next
            current['end_time'] = max(current['end_time'], next_punch['end_time'])
            current['end_frame'] = max(current['end_frame'], next_punch['end_frame'])
            current['duration'] = current['end_time'] - current['start_time']
            current['confidence'] = max(current['confidence'], next_punch['confidence'])
            
            # Prefer jab label if both are present
            if prefer_jab:
                if 'punch_type' in current and current['punch_type'] == 'jab':
                    # Keep jab label
                    pass
                elif 'punch_type' in next_punch and next_punch['punch_type'] == 'jab':
                    # Switch to jab label
                    current['punch_type'] = 'jab'
                    if 'hand' in next_punch:
                        current['hand'] = next_punch['hand']
        else:
            # No overlap - save current and start new
            merged.append(current)
            current = next_punch.copy()

    merged.append(current)
    return merged


def merge_overlapping_jabs(jab_windows, overlap_threshold=0.5):
    """Merge overlapping jab detections (legacy function for backward compatibility)."""
    if len(jab_windows) == 0:
        return []

    # Convert DataFrame to list of dicts if needed
    if isinstance(jab_windows, pd.DataFrame):
        jabs = jab_windows.sort_values('start_time').to_dict('records')
    else:
        jabs = sorted(jab_windows, key=lambda x: x['start_time'])
    
    merged = []
    current = jabs[0].copy()

    for next_jab in jabs[1:]:
        # Check if overlapping
        overlap_start = max(current['start_time'], next_jab['start_time'])
        overlap_end = min(current['end_time'], next_jab['end_time'])
        overlap_duration = max(0, overlap_end - overlap_start)

        min_duration = min(current['duration'], next_jab['duration'])

        if overlap_duration / min_duration >= overlap_threshold:
            # Merge - extend current to include next
            current['end_time'] = max(current['end_time'], next_jab['end_time'])
            current['end_frame'] = max(current['end_frame'], next_jab['end_frame'])
            current['duration'] = current['end_time'] - current['start_time']
            current['confidence'] = max(current['confidence'], next_jab['confidence'])
        else:
            # No overlap - save current and start new
            merged.append(current)
            current = next_jab.copy()

    merged.append(current)
    return merged


def windows_overlap(start1, end1, start2, end2, threshold=WINDOW_OVERLAP_THRESHOLD):
    """
    Check if two frame ranges overlap by at least threshold fraction.

    Args:
        start1, end1: First range (start_frame, end_frame)
        start2, end2: Second range (start_frame, end_frame)
        threshold: Minimum overlap ratio (default from config: 0.3 = 30% overlap)

    Returns:
        True if overlap >= threshold, False otherwise

    Example:
        windows_overlap(40, 55, 50, 65, 0.3) -> True (overlap of 5 frames over 15-frame min range = 33%)
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap = max(0, overlap_end - overlap_start)

    min_duration = min(end1 - start1, end2 - start2)
    if min_duration == 0:
        return False

    return (overlap / min_duration) >= threshold


def calculate_comprehensive_metrics(merged_jabs, landmark_df, fps=15):
    """
    Calculate comprehensive session metrics from detected jabs and landmark data.

    Returns a dictionary with:
    - Performance metrics (max/min/avg speed, variance)
    - Temporal metrics (duration, active/rest time, activity %)
    - Intensity metrics (bursts, intervals)
    - Fatigue indicators
    """
    total_punches = len(merged_jabs)

    # Get video duration from landmark data
    total_frames = len(landmark_df)
    session_duration = float(total_frames / fps)

    # Initialize metrics
    metrics = {
        'session_duration': session_duration,
        'total_punches': int(total_punches),
    }

    if total_punches == 0:
        # Return empty metrics for sessions with no punches
        metrics.update({
            'average_speed': 0.0,
            'max_speed': 0.0,
            'min_speed': 0.0,
            'speed_variance': 0.0,
            'punches_per_minute': 0.0,
            'average_punch_duration': 0.0,
            'active_time': 0.0,
            'rest_time': session_duration,
            'activity_percentage': 0.0,
            'average_rest_period': 0.0,
            'max_rest_period': session_duration,
            'min_rest_period': 0.0,
            'punch_intensity_score': 0.0,
            'fatigue_indicator': 0.0,
            'speed_trend': 'stable',
            'punches_by_interval': [],
            'peak_interval': None,
            'fastest_punch': None,
            'slowest_punch': None,
        })
        return metrics

    # Extract speeds and durations
    speeds = [jab['confidence'] * 5 for jab in merged_jabs]  # Scale confidence to speed
    durations = [jab['duration'] for jab in merged_jabs]
    start_times = [jab['start_time'] for jab in merged_jabs]
    end_times = [jab['end_time'] for jab in merged_jabs]

    # Performance metrics
    metrics['average_speed'] = float(np.mean(speeds))
    metrics['max_speed'] = float(np.max(speeds))
    metrics['min_speed'] = float(np.min(speeds))
    metrics['speed_variance'] = float(np.std(speeds))
    metrics['average_punch_duration'] = float(np.mean(durations))

    # Find fastest and slowest punches
    fastest_idx = int(np.argmax(speeds))
    slowest_idx = int(np.argmin(speeds))

    metrics['fastest_punch'] = {
        'index': fastest_idx + 1,
        'speed': float(speeds[fastest_idx]),
        'time': float(start_times[fastest_idx]),
        'duration': float(durations[fastest_idx])
    }

    metrics['slowest_punch'] = {
        'index': slowest_idx + 1,
        'speed': float(speeds[slowest_idx]),
        'time': float(start_times[slowest_idx]),
        'duration': float(durations[slowest_idx])
    }

    # Temporal/Activity metrics
    active_time = float(sum(durations))
    rest_time = float(session_duration - active_time)

    metrics['active_time'] = active_time
    metrics['rest_time'] = rest_time
    metrics['activity_percentage'] = float((active_time / session_duration * 100) if session_duration > 0 else 0)
    metrics['punches_per_minute'] = float((total_punches / session_duration * 60) if session_duration > 0 else 0)

    # Rest periods between punches
    rest_periods = []
    for i in range(len(merged_jabs) - 1):
        rest_period = merged_jabs[i+1]['start_time'] - merged_jabs[i]['end_time']
        rest_periods.append(max(0, rest_period))

    if rest_periods:
        metrics['average_rest_period'] = float(np.mean(rest_periods))
        metrics['max_rest_period'] = float(np.max(rest_periods))
        metrics['min_rest_period'] = float(np.min(rest_periods))
    else:
        metrics['average_rest_period'] = 0.0
        metrics['max_rest_period'] = 0.0
        metrics['min_rest_period'] = 0.0

    # Intensity metrics - punches per 10-second interval
    num_intervals = int(np.ceil(session_duration / 10))
    punches_by_interval = [0] * num_intervals

    for jab in merged_jabs:
        interval_idx = int(jab['start_time'] // 10)
        if interval_idx < num_intervals:
            punches_by_interval[interval_idx] += 1

    metrics['punches_by_interval'] = [int(x) for x in punches_by_interval]
    metrics['peak_interval'] = {
        'interval': int(np.argmax(punches_by_interval)),
        'punches': int(np.max(punches_by_interval))
    }

    # Punch intensity score (combines frequency and speed)
    intensity_score = (metrics['punches_per_minute'] / 60) * metrics['average_speed']
    metrics['punch_intensity_score'] = float(intensity_score)

    # Fatigue analysis - compare first half vs second half
    midpoint = session_duration / 2
    first_half_jabs = [jab for jab in merged_jabs if jab['start_time'] < midpoint]
    second_half_jabs = [jab for jab in merged_jabs if jab['start_time'] >= midpoint]

    if first_half_jabs and second_half_jabs:
        first_half_speeds = [jab['confidence'] * 5 for jab in first_half_jabs]
        second_half_speeds = [jab['confidence'] * 5 for jab in second_half_jabs]

        avg_speed_first = float(np.mean(first_half_speeds))
        avg_speed_second = float(np.mean(second_half_speeds))

        # Fatigue indicator: negative means slower in second half
        speed_change_percent = ((avg_speed_second - avg_speed_first) / avg_speed_first * 100) if avg_speed_first > 0 else 0
        metrics['fatigue_indicator'] = float(speed_change_percent)

        if speed_change_percent < -10:
            metrics['speed_trend'] = 'declining'
        elif speed_change_percent > 10:
            metrics['speed_trend'] = 'improving'
        else:
            metrics['speed_trend'] = 'stable'
    else:
        metrics['fatigue_indicator'] = 0.0
        metrics['speed_trend'] = 'stable'

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run jab detection inference on test videos")
    parser.add_argument('--video_name', type=str, default=None,
                        help='Specific video to analyze (e.g., IMG_0759). If not provided, analyzes all test videos.')
    args = parser.parse_args()

    # Load model
    model_path = MODELS_DIR / "jab_classifier.joblib"
    feature_names_path = MODELS_DIR / "feature_names.txt"

    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("Please train the model first using: python -m src.train_jab_model")
        return

    print("Loading trained model...")
    model = joblib.load(model_path)

    # Load feature names
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]

    print(f"Model loaded successfully")
    print(f"Features: {len(feature_names)}")

    # Find test landmark files
    landmark_files = list(LANDMARKS_DIR.glob("test_*.csv"))

    if args.video_name:
        # Filter to specific video
        landmark_files = [f for f in landmark_files if args.video_name in f.name]

    if not landmark_files:
        print(f"\n[ERROR] No test landmark files found in {LANDMARKS_DIR}")
        print("Please run pose extraction first:")
        print("  python -m src.pose_extraction --input_dir data/test/videos --label test")
        return

    print(f"\nFound {len(landmark_files)} test video(s)")
    print("=" * 60)

    # Analyze each video
    all_results = {}

    for landmark_file in landmark_files:
        jab_windows = analyze_video(landmark_file, model, feature_names)

        if len(jab_windows) > 0:
            # Merge overlapping detections
            merged_jabs = merge_overlapping_jabs(jab_windows)
            all_results[landmark_file.stem] = merged_jabs

            print(f"\n  [RESULTS] Detected {len(merged_jabs)} jab event(s):")
            for i, jab in enumerate(merged_jabs, 1):
                print(f"     {i}. Time: {jab['start_time']:.2f}s - {jab['end_time']:.2f}s "
                      f"(duration: {jab['duration']:.2f}s, confidence: {jab['confidence']:.2%})")
        else:
            print(f"  [INFO] No jabs detected")
            all_results[landmark_file.stem] = []

    print("\n" + "=" * 60)
    print("Analysis complete!")

    # Summary
    total_jabs = sum(len(jabs) for jabs in all_results.values())
    print(f"\n[SUMMARY]")
    print(f"   Videos analyzed: {len(all_results)}")
    print(f"   Total jabs detected: {total_jabs}")


if __name__ == "__main__":
    main()
