import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from django.conf import settings
from ultralytics import YOLO

try:
    from picamzero import Camera  # type: ignore
    PICAMZERO_AVAILABLE = True
except ImportError:
    PICAMZERO_AVAILABLE = False

MODEL_URL = "https://www.dropbox.com/scl/fi/8n60aqre52ix3gp65t3v0/best.onnx?rlkey=1jniqjxlctut2qopgagsr6lkm&st=ftgl9dsj&dl=1"
MODEL_FILENAME = "best.onnx"
MODEL_DIR = settings.BASE_DIR / "content" / "runs" / "detect" / "parking_model" / "weights"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

_model: Optional[YOLO] = None


def _download_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return

    response = requests.get(MODEL_URL, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 256

    downloaded = 0
    try:
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size):
                if not chunk:
                    continue
                file.write(chunk)
                downloaded += len(chunk)
                if total_size and downloaded >= total_size:
                    break
    except Exception:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise


def get_model() -> YOLO:
    global _model
    if _model is None:
        _download_model()
        _model = YOLO(str(MODEL_PATH), task="detect")
    return _model


def _decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img


def run_inference(image_bytes: bytes, conf: float = 0.25) -> bytes:
    model = get_model()
    frame = _decode_image(image_bytes)
    results = model.predict(source=frame, conf=conf)
    plotted = results[0].plot()
    ok, encoded = cv2.imencode(".jpg", plotted)
    if not ok:
        raise RuntimeError("Failed to encode result image")
    return encoded.tobytes()


def camera_available() -> bool:
    if PICAMZERO_AVAILABLE:
        try:
            Camera()
            return True
        except Exception:
            return False

    camera = cv2.VideoCapture(0)
    ready = camera.isOpened()
    camera.release()
    return ready


def capture_frame() -> bytes:
    if PICAMZERO_AVAILABLE:
        temp_path = Path(tempfile.gettempdir()) / "picamzero_capture.jpg"
        cam = Camera()
        try:
            cam.start_preview()
            cam.take_photo(str(temp_path))
        finally:
            try:
                cam.stop_preview()
            except Exception:
                pass

        if not temp_path.exists():
            raise RuntimeError("Picamzero failed to create capture file.")

        data = temp_path.read_bytes()
        temp_path.unlink(missing_ok=True)
        if not data:
            raise RuntimeError("Captured file is empty.")
        return data

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("No camera detected. Connect a Raspberry Pi camera or USB webcam.")

    ok, frame = camera.read()
    camera.release()
    if not ok:
        raise RuntimeError("Camera found but failed to capture a frame.")

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode captured frame.")
    return encoded.tobytes()


def as_data_uri(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
