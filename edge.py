"""Dependency setup for OpenCV-based edge processing.

Install dependency with:
    pip install opencv-python
"""

try:
    import cv2
except ImportError as exc:
    raise SystemExit("Missing dependency. Run: pip install opencv-python") from exc
