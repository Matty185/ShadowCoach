"""
Quick test script to verify the optimized detection pipeline and measure performance improvements.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import LANDMARKS_DIR, MODELS_DIR, WINDOW_SIZE_FRAMES, WINDOW_STEP_FRAMES
from inference import (
    compute_per_frame_features,
    analyze_video,  # Legacy
    analyze_video_optimized,  # Optimized
    merge_overlapping_jabs,
    merge_overlapping_punches
)

def test_optimized_pipeline():
    """Test the optimized pipeline and compare with legacy."""

    print("=" * 70)
    print("OPTIMIZED DETECTION PIPELINE TEST")
    print("=" * 70)

    # Load model
    model_path = MODELS_DIR / "jab_classifier.joblib"
    feature_names_path = MODELS_DIR / "feature_names.txt"

    if not model_path.exists():
        print("[ERROR] Model not found. Please train the model first.")
        return

    print("\n[1/5] Loading model...")
    model = joblib.load(model_path)

    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]

    print(f"   [OK] Model loaded with {len(feature_names)} features")

    # Find a test video
    test_files = list(LANDMARKS_DIR.glob("test_*.csv"))

    if not test_files:
        print("[ERROR] No test landmark files found.")
        return

    landmark_file = test_files[0]  # Use first test video
    print(f"\n[2/5] Loading test video: {landmark_file.name}")

    # Load data
    landmark_df = pd.read_csv(landmark_file)
    feature_df = compute_per_frame_features(landmark_df)

    total_frames = len(feature_df)
    total_possible_windows = max(1, (total_frames - WINDOW_SIZE_FRAMES + 1) // WINDOW_STEP_FRAMES)

    print(f"   [OK] Loaded {total_frames} frames")
    print(f"   [OK] Total possible windows: {total_possible_windows}")

    # Test LEGACY pipeline
    print(f"\n[3/5] Running LEGACY pipeline...")
    start_time = time.time()

    jab_windows = analyze_video(landmark_file, model, feature_names, fps=15)
    legacy_time = time.time() - start_time

    merged_jabs_legacy = merge_overlapping_jabs(jab_windows) if len(jab_windows) > 0 else []

    for jab in merged_jabs_legacy:
        jab['punch_type'] = 'jab'

    print(f"   [OK] Legacy completed in {legacy_time:.3f}s")
    print(f"   [OK] Windows processed: {total_possible_windows}")
    print(f"   [OK] Jabs detected: {len(merged_jabs_legacy)}")

    # Test OPTIMIZED pipeline
    print(f"\n[4/5] Running OPTIMIZED pipeline...")
    start_time = time.time()

    ml_classified, unclassified = analyze_video_optimized(
        feature_df=feature_df,
        landmark_df=landmark_df,
        model=model,
        feature_names=feature_names,
        fps=15
    )

    optimized_time = time.time() - start_time

    all_punches = ml_classified + unclassified
    merged_punches_optimized = merge_overlapping_punches(all_punches, prefer_jab=True) if len(all_punches) > 0 else []

    print(f"   [OK] Optimized completed in {optimized_time:.3f}s")
    print(f"   [OK] Jabs detected: {len(ml_classified)}")
    print(f"   [OK] Generic punches detected: {len(unclassified)}")
    print(f"   [OK] Total punches: {len(merged_punches_optimized)}")

    # Performance comparison
    print(f"\n[5/5] PERFORMANCE COMPARISON")
    print("=" * 70)

    if legacy_time > 0:
        speedup = legacy_time / optimized_time
        time_saved = legacy_time - optimized_time
        print(f"   Legacy time:     {legacy_time:.3f}s")
        print(f"   Optimized time:  {optimized_time:.3f}s")
        print(f"   Time saved:      {time_saved:.3f}s ({(time_saved/legacy_time*100):.1f}% faster)")
        print(f"   Speedup factor:  {speedup:.2f}x")

    # Detection comparison
    print(f"\n   Detection Results:")
    print(f"   - Legacy jabs:     {len(merged_jabs_legacy)}")
    print(f"   - Optimized total: {len(merged_punches_optimized)} ({len(ml_classified)} jabs + {len(unclassified)} generic)")

    # Summary
    print("\n" + "=" * 70)
    if len(merged_punches_optimized) >= len(merged_jabs_legacy):
        print("[SUCCESS] Optimized pipeline detects same or more punches")
    else:
        print("[WARNING] Optimized pipeline detected fewer punches")

    print("=" * 70)

    return {
        'legacy_time': legacy_time,
        'optimized_time': optimized_time,
        'legacy_detections': len(merged_jabs_legacy),
        'optimized_detections': len(merged_punches_optimized),
        'speedup': speedup if legacy_time > 0 else 0
    }


if __name__ == "__main__":
    try:
        results = test_optimized_pipeline()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
