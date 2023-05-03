from django.contrib import admin
from django.urls import path, include
from startyolo import views

urlpatterns = [
    path('', views.index, name = 'index'),
    path('video', views.video, name = 'video'),
]
