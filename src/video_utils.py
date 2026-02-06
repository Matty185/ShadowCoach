"""
Utilities for video frame extraction and processing.
"""
import cv2
from pathlib import Path
from typing import Iterator, Tuple
import numpy as np


def iter_video_frames(video_path: str, target_fps: int) -> Iterator[Tuple[int, float, np.ndarray]]:
    """
    Yields (frame_idx, timestamp_sec, frame_bgr) for the given video.
    Downsamples to approximately target_fps if the source FPS is higher.

    Args:
        video_path: Path to the video file
        target_fps: Target frames per second to extract

    Yields:
        Tuple of (frame_index, timestamp_seconds, frame_bgr_array)

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot be opened
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    try:
        # Get video properties
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            raise ValueError(f"Invalid FPS detected in video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / source_fps if source_fps > 0 else 0
        
        # Calculate frame stride for downsampling
        if source_fps > target_fps:
            stride = max(1, int(round(source_fps / target_fps)))
        else:
            stride = 1
        
        frame_idx = 0
        output_frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Only yield frames at target FPS
            if frame_idx % stride == 0:
                timestamp_sec = frame_idx / source_fps
                yield output_frame_idx, timestamp_sec, frame
                output_frame_idx += 1
            
            frame_idx += 1
    
    finally:
        cap.release()


if __name__ == "__main__":
    """
    Test script for video frame extraction.
    Usage: python -m src.video_utils [video_path]
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Test video frame extraction")
    parser.add_argument(
        "video_path",
        nargs="?",
        type=str,
        help="Path to video file to test (optional, will use first video in jab dir if not provided)"
    )
    
    args = parser.parse_args()
    
    # If no video path provided, try to find one in the jab directory
    if not args.video_path:
        from src.config import RAW_JAB_DIR
        video_files = list(RAW_JAB_DIR.glob("*.mp4"))
        if not video_files:
            print(f"No video files found in {RAW_JAB_DIR}")
            print("Usage: python -m src.video_utils <video_path>")
            sys.exit(1)
        args.video_path = str(video_files[0])
        print(f"Using test video: {args.video_path}")
    
    try:
        from src.config import TARGET_FPS
        
        frame_count = 0
        first_timestamp = None
        last_timestamp = None
        
        print(f"Extracting frames from: {args.video_path}")
        print(f"Target FPS: {TARGET_FPS}")
        print("-" * 50)
        
        for frame_idx, timestamp_sec, frame in iter_video_frames(args.video_path, TARGET_FPS):
            if first_timestamp is None:
                first_timestamp = timestamp_sec
            last_timestamp = timestamp_sec
            frame_count += 1
            
            if frame_count <= 5:  # Print first 5 frames
                print(f"Frame {frame_idx}: timestamp={timestamp_sec:.3f}s, shape={frame.shape}")
        
        print("-" * 50)
        print(f"Total frames extracted: {frame_count}")
        if last_timestamp is not None:
            print(f"Duration: {last_timestamp:.2f} seconds")
            print(f"Effective FPS: {frame_count / last_timestamp:.2f}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

