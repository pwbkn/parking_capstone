from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from .models import ParkingSpot, ParkingZone, AnalyticsData
from .yolo_service import MODEL_PATH, run_inference, camera_available, capture_frame, as_data_uri
import random # Simulating ML data for demo

@login_required
def dashboard_view(request):
    # Determine stats
    total_spots = ParkingSpot.objects.count()
    occupied = ParkingSpot.objects.filter(is_occupied=True).count()
    
    # Calculate percentage for the donut chart
    occupancy_rate = int((occupied / total_spots) * 100) if total_spots > 0 else 0
    
    # Context data for the template
    context = {
        'total_spots': total_spots,
        'occupied_spots': occupied,
        'free_spots': total_spots - occupied,
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
                annotated = run_inference(captured_bytes)
                result_image = as_data_uri(annotated)
                source = 'capture'
            except Exception as exc: # noqa: B902 (broad for user feedback)
                error = str(exc)
        elif request.FILES.get('image'):
            try:
                uploaded_bytes = request.FILES['image'].read()
                annotated = run_inference(uploaded_bytes)
                result_image = as_data_uri(annotated)
                source = 'upload'
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