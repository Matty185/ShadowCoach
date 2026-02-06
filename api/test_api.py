"""
Test script for the API.
Tests the /analyze endpoint with a test video.
"""

import requests
from pathlib import Path

# API endpoint
API_URL = "http://localhost:5000/analyze"

# Test video path
TEST_VIDEO = Path(__file__).parent.parent / "data" / "test" / "videos" / "IMG_0756.mov"

def test_health():
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get("http://localhost:5000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_analyze():
    """Test the /analyze endpoint with a video."""
    if not TEST_VIDEO.exists():
        print(f"Test video not found: {TEST_VIDEO}")
        return

    print(f"Testing /analyze endpoint with: {TEST_VIDEO.name}")
    print(f"File size: {TEST_VIDEO.stat().st_size / (1024*1024):.2f} MB")
    print()

    with open(TEST_VIDEO, 'rb') as f:
        files = {'video': (TEST_VIDEO.name, f, 'video/quicktime')}
        print("Uploading video...")
        response = requests.post(API_URL, files=files)

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\n=== Analysis Results ===")
        print(f"Total punches: {result['total_punches']}")
        print(f"Average speed: {result['average_speed']:.2f}")
        print(f"Punches per minute: {result['punches_per_minute']:.2f}")
        print(f"\nPunch events:")
        for punch in result['punch_events']:
            print(f"  {punch['index']}. Time: {punch['start']:.2f}s - {punch['end']:.2f}s (speed: {punch['speed']:.2f})")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Shadow Coach API Test")
    print("=" * 60)
    print()

    try:
        test_health()
        test_analyze()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server.")
        print("Make sure the server is running: python api/app.py")
    except Exception as e:
        print(f"Error: {e}")
