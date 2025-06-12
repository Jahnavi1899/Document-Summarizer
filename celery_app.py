# celery_app.py
from celery import Celery
import os

# Ensure environment variables are loaded if using python-dotenv
from dotenv import load_dotenv
load_dotenv()

# Configure Celery to use Redis as the broker and result backend
REDIS_BROKER_URL = os.getenv("CELERY_BROKER_URL")
REDIS_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")

celery_app = Celery(
    'app_backend', # Replace with a suitable name for your app
    broker=REDIS_BROKER_URL,
    backend=REDIS_RESULT_BACKEND,
    # Celery needs to know where to find your task functions
    include=['app_backend.tasks'] # Adjust based on your actual package name
)

# Optional: Celery configuration settings
celery_app.conf.update(
    task_track_started=True, # Track tasks as "STARTED" state
    task_acks_late=True,     # Task is acknowledged after it's done (more robust)
    worker_prefork_reachable=False # Avoids a warning with latest Celery/Python 3.10+
)

if __name__ == '__main__':
    celery_app.start()