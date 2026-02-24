from django.urls import path
from . import views

urlpatterns = [
    path("",          views.index,   name="index"),
    path("analyze/",  views.analyze, name="analyze"),
    path("result/<int:pk>/", views.result, name="result"),
    path("history/",  views.history, name="history"),
]
