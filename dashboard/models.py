from django.db import models
from django.contrib.auth.models import User

class ParkingZone(models.Model):
    name = models.CharField(max_length=50) # e.g., "Zone A"
    total_spots = models.IntegerField(default=0)
    
    def __str__(self):
        return self.name

class ParkingSpot(models.Model):
    spot_number = models.CharField(max_length=10) # e.g., "203"
    zone = models.ForeignKey(ParkingZone, on_delete=models.CASCADE)
    is_occupied = models.BooleanField(default=False)
    last_updated = models.DateTimeField(auto_now=True)
    sensor_id = models.CharField(max_length=50, blank=True, null=True) # For RPi integration

    def __str__(self):
        return f"{self.spot_number} - {'Occupied' if self.is_occupied else 'Free'}"

class AnalyticsData(models.Model):
    # Stores historical data for the line chart
    timestamp = models.DateTimeField(auto_now_add=True)
    occupancy_rate = models.IntegerField() # 0-100
    accuracy_score = models.FloatField(default=0.0)