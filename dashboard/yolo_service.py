import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from django.conf import settings
from ultralytics import YOLO

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


def _extract_occupancy_stats(result) -> Dict[str, int]:
    """Count occupied/empty detections from a YOLO result."""
    counts = {"occupied": 0, "empty": 0, "lots": 0}

    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    class_ids = boxes.cls.tolist() if boxes is not None and boxes.cls is not None else []

    for cid in class_ids:
        label = str(names.get(int(cid), "")).lower()
        if "empty" in label:
            counts["empty"] += 1
        elif "occup" in label:
            counts["occupied"] += 1
        elif "lot" in label:
            counts["lots"] += 1

    total_spaces = counts["occupied"] + counts["empty"]
    occupancy_rate = int(round((counts["occupied"] / total_spaces) * 100)) if total_spaces else 0

    return {
        "occupied": counts["occupied"],
        "empty": counts["empty"],
        "total_spaces": total_spaces,
        "occupancy_rate": occupancy_rate,
        "lots_detected": counts["lots"],
    }


def run_inference(image_bytes: bytes, conf: float = 0.25) -> Tuple[bytes, Dict[str, int]]:
    model = get_model()
    frame = _decode_image(image_bytes)
    results = model.predict(source=frame, conf=conf)
    result = results[0]
    stats = _extract_occupancy_stats(result)
    plotted = result.plot()
    ok, encoded = cv2.imencode(".jpg", plotted)
    if not ok:
        raise RuntimeError("Failed to encode result image")
    return encoded.tobytes(), stats


def camera_available() -> bool:
    rpicam_exists = shutil.which("rpicam-still") is not None
    if rpicam_exists or shutil.which("fswebcam"):
        return True

    for device in (0, 1):
        camera = cv2.VideoCapture(device)
        ready = camera.isOpened()
        camera.release()
        if ready:
            return True
    return False


def _capture_with_webcam() -> bytes:
    errors = []
    for device in (0, 1):
        camera = cv2.VideoCapture(device)
        if not camera.isOpened():
            camera.release()
            errors.append(f"USB webcam not detected on /dev/video{device}.")
            continue

        ok, frame = camera.read()
        camera.release()
        if not ok:
            errors.append(f"USB webcam found on /dev/video{device} but failed to capture a frame.")
            continue

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            errors.append(f"Failed to encode captured frame from /dev/video{device}.")
            continue
        return encoded.tobytes()

    raise RuntimeError("; ".join(errors) if errors else "No USB webcam detected.")


def _capture_with_fswebcam() -> bytes:
    exe = shutil.which("fswebcam")
    if not exe:
        raise RuntimeError("fswebcam is not installed.")

    errors = []
    for device in ("/dev/video1", "/dev/video0"):
        temp_path = Path(tempfile.gettempdir()) / "fswebcam_capture.jpg"
        cmd = [exe, "-d", device, "-r", "1280x720", "--no-banner", "-S", "20", str(temp_path)]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=12)
            if proc.returncode != 0:
                errors.append(f"{device}: {proc.stderr.decode().strip() or 'fswebcam failed'}")
                continue
            if not temp_path.exists():
                errors.append(f"{device}: fswebcam did not create an output file.")
                continue

            data = temp_path.read_bytes()
            if not data:
                errors.append(f"{device}: fswebcam produced an empty file.")
                continue
            return data
        except subprocess.TimeoutExpired:
            errors.append(f"{device}: fswebcam timed out after 12 seconds")
        finally:
            temp_path.unlink(missing_ok=True)

    raise RuntimeError("; ".join(errors) if errors else "fswebcam could not capture on any device")


def capture_frame() -> bytes:
    errors = []
    rpicam = shutil.which("rpicam-still")
    if rpicam:
        temp_path = Path(tempfile.gettempdir()) / "rpicam_capture.jpg"
        cmd = [rpicam, "-o", str(temp_path), "-t", "1000"]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            if proc.returncode == 0 and temp_path.exists():
                data = temp_path.read_bytes()
                temp_path.unlink(missing_ok=True)
                if not data:
                    errors.append("rpicam-still produced an empty file.")
                else:
                    return data
            else:
                err_msg = proc.stderr.decode().strip() or "rpicam-still failed with unknown error"
                errors.append(err_msg)
        except subprocess.TimeoutExpired:
            errors.append("rpicam-still timed out after 10 seconds")
        except Exception as exc:  # noqa: B902
            errors.append(f"rpicam-still error: {exc}")
        finally:
            temp_path.unlink(missing_ok=True)

    try:
        return _capture_with_fswebcam()
    except Exception as exc:  # noqa: B902
        errors.append(str(exc))

    try:
        return _capture_with_webcam()
    except Exception as exc:  # noqa: B902
        errors.append(str(exc))

    raise RuntimeError("No camera could capture a frame. Attempts: " + "; ".join(errors))


def as_data_uri(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
