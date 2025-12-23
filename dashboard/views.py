import base64
import csv
import json
from pathlib import Path

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.http import JsonResponse
from django.utils import timezone
from django.conf import settings
from .models import ParkingSpot, ParkingZone, AnalyticsData
from .yolo_service import MODEL_PATH, run_inference, camera_available, capture_frame, as_data_uri
import random # Simulating ML data for demo


def _persist_scan(stats: dict, source: str) -> None:
    """Store a historical record of a scan for analytics."""
    try:
        AnalyticsData.objects.create(
            occupancy_rate=int(stats.get("occupancy_rate", 0)),
            accuracy_score=float(stats.get("accuracy_score", 0.0)),
            total_spaces=int(stats.get("total_spaces", 0)),
            occupied_spaces=int(stats.get("occupied", 0)),
            free_spaces=int(stats.get("empty", 0)),
            source=source,
        )
    except Exception:
        # Do not block the request if analytics logging fails.
        pass


def _store_latest_stats(stats: dict, source: str) -> None:
    """Persist the most recent occupancy stats for the dashboard card and history."""
    payload = {
        **stats,
        "source": source,
        "updated_at": timezone.now().isoformat(),
    }
    cache.set("latest_occupancy_stats", payload, timeout=None)
    _persist_scan(stats, source)

@login_required
def dashboard_view(request):
    # Determine stats
    total_spots = ParkingSpot.objects.count()
    occupied = ParkingSpot.objects.filter(is_occupied=True).count()
    
    # Calculate percentage for the donut chart
    occupancy_rate = int((occupied / total_spots) * 100) if total_spots > 0 else 0

    latest_stats = cache.get("latest_occupancy_stats")
    if latest_stats:
        total_spots = latest_stats.get("total_spaces", total_spots)
        occupied = latest_stats.get("occupied", occupied)
        free = latest_stats.get("empty", total_spots - occupied)
        occupancy_rate = latest_stats.get("occupancy_rate", occupancy_rate)
    else:
        free = total_spots - occupied
    
    # Context data for the template
    context = {
        'total_spots': total_spots,
        'occupied_spots': occupied,
        'free_spots': free,
        'occupancy_rate': occupancy_rate,
        'active_segment': 'Dashboard'
    }
    return render(request, 'dashboard.html', context)


@login_required
def cameras_view(request):
    result_image = None
    error = None
    source = None
    camera_ready = camera_available()

    if request.method == 'POST':
        if 'capture' in request.POST:
            try:
                captured_bytes = capture_frame()
                annotated, stats = run_inference(captured_bytes)
                result_image = as_data_uri(annotated)
                source = 'capture'
                _store_latest_stats(stats, source)
            except Exception as exc: # noqa: B902 (broad for user feedback)
                error = str(exc)
        elif request.FILES.get('image'):
            try:
                uploaded_bytes = request.FILES['image'].read()
                annotated, stats = run_inference(uploaded_bytes)
                result_image = as_data_uri(annotated)
                source = 'upload'
                _store_latest_stats(stats, source)
            except Exception as exc: # noqa: B902
                error = str(exc)
        else:
            error = "Please upload an image or use the capture option."

    context = {
        'active_segment': 'Cameras',
        'camera_ready': camera_ready,
        'result_image': result_image,
        'error': error,
        'source': source,
        'MODEL_PATH': MODEL_PATH,
    }
    return render(request, 'cameras.html', context)

def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def get_stats(request):
    # API for dynamic charts (simulated data for the line chart)
    data = {
        'labels': ['10:00', '11:00', '12:00', '13:00', '14:00'],
        'data': [10, 25, 60, 40, 75] 
    }
    return JsonResponse(data)


@login_required
def analytics_view(request):
    history = list(AnalyticsData.objects.all())
    # For charting we want chronological order
    chart_points = list(reversed(history))

    labels = [item.timestamp.strftime("%b %d, %H:%M") for item in chart_points]
    occupied = [item.occupied_spaces for item in chart_points]
    total = [item.total_spaces for item in chart_points]
    free = [item.free_spaces for item in chart_points]

    total_scans = len(history)
    avg_occupancy = int(round(sum(item.occupancy_rate for item in history) / total_scans)) if total_scans else 0
    last_scan = history[0] if history else None
    peak_occupied = max(occupied) if occupied else 0

    chart_payload = {
        "labels": labels,
        "occupied": occupied,
        "total": total,
        "free": free,
    }

    context = {
        "active_segment": "Analytics",
        "history": history,
        "chart_json": json.dumps(chart_payload),
        "total_scans": total_scans,
        "avg_occupancy": avg_occupancy,
        "peak_occupied": peak_occupied,
        "last_scan": last_scan,
    }
    return render(request, "analytics.html", context)


@login_required
def training_results_view(request):
    csv_path = Path(settings.BASE_DIR) / "results_training" / "results.csv"
    rows = []
    headers = []
    chart_payload = {"labels": [], "map50": [], "map5095": []}
    table_rows = []
    images = []
    summary = {
        "epochs": 0,
        "best_map50": None,
        "best_map5095": None,
        "last_map50": None,
        "last_map5095": None,
        "last_precision": None,
        "last_recall": None,
        "last_epoch": None,
        "total_time": 0.0,
    }

    if csv_path.exists():
        with csv_path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            for row in reader:
                try:
                    row_num = {k: float(v) if k != "epoch" else int(float(v)) for k, v in row.items()}
                except ValueError:
                    row_num = row
                rows.append(row_num)

        if rows:
            summary["epochs"] = len(rows)
            last = rows[-1]
            summary["last_epoch"] = last.get("epoch")
            summary["last_map50"] = last.get("metrics/mAP50(B)")
            summary["last_map5095"] = last.get("metrics/mAP50-95(B)")
            summary["last_precision"] = last.get("metrics/precision(B)")
            summary["last_recall"] = last.get("metrics/recall(B)")
            summary["total_time"] = sum(r.get("time", 0) for r in rows if isinstance(r, dict))

            summary["best_map50"] = max((r.get("metrics/mAP50(B)") for r in rows), default=None)
            summary["best_map5095"] = max((r.get("metrics/mAP50-95(B)") for r in rows), default=None)

            chart_payload["labels"] = [r.get("epoch") for r in rows]
            chart_payload["map50"] = [r.get("metrics/mAP50(B)") for r in rows]
            chart_payload["map5095"] = [r.get("metrics/mAP50-95(B)") for r in rows]

            if headers:
                table_rows = [[r.get(h, "") for h in headers] for r in rows]

    # Load a curated set of result images as data URIs for the gallery
    image_candidates = [
        ("Confusion Matrix", "confusion_matrix.png"),
        ("Normalized Confusion", "confusion_matrix_normalized.png"),
        ("PR Curve", "BoxPR_curve.png"),
        ("P Curve", "BoxP_curve.png"),
        ("R Curve", "BoxR_curve.png"),
        ("F1 Curve", "BoxF1_curve.png"),
        ("Results Summary", "results.png"),
        ("Label Distribution", "labels.jpg"),
        ("Val Predictions 0", "val_batch0_pred.jpg"),
        ("Val Predictions 1", "val_batch1_pred.jpg"),
        ("Val Labels 0", "val_batch0_labels.jpg"),
        ("Val Labels 1", "val_batch1_labels.jpg"),
        ("Train Batch 0", "train_batch0.jpg"),
        ("Train Batch 1", "train_batch1.jpg"),
        ("Train Batch 2", "train_batch2.jpg"),
        ("Train Batch 1560", "train_batch1560.jpg"),
        ("Train Batch 1561", "train_batch1561.jpg"),
        ("Train Batch 1562", "train_batch1562.jpg"),
    ]

    for title, fname in image_candidates:
        fpath = csv_path.parent / fname
        if not fpath.exists():
            continue
        try:
            data = fpath.read_bytes()
            ext = fpath.suffix.lower().lstrip(".") or "png"
            b64 = base64.b64encode(data).decode("ascii")
            images.append({
                "title": title,
                "src": f"data:image/{ext};base64,{b64}",
                "name": fname,
            })
        except Exception:
            continue

    context = {
        "active_segment": "TrainingResults",
        "headers": headers,
        "rows": rows,
        "table_rows": table_rows,
        "chart_json": json.dumps(chart_payload),
        "summary": summary,
        "csv_missing": not csv_path.exists(),
        "csv_path": str(csv_path),
        "images": images,
    }
    return render(request, "training_results.html", context)