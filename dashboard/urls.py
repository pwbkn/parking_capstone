from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('cameras/', views.cameras_view, name='cameras'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('training-results/', views.training_results_view, name='training_results'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    # API endpoint for the charts (Optional but professional)
    path('api/stats/', views.get_stats, name='get_stats'),
]