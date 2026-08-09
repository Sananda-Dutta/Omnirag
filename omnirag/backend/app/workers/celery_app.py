"""
Celery application instance.

Why Celery+Redis over FastAPI's built-in BackgroundTasks:
    BackgroundTasks run in the same process as the API, after the response
    is sent. That's fine for genuinely trivial work, but document extraction
    (parsing a 200-page PDF) can take real time and real CPU — doing that in
    the API process means a burst of uploads degrades response time for
    every *other* request the same process is trying to serve. Celery
    workers are separate processes (separately scalable, separately
    restartable if a worker crashes on a malformed file) and Redis gives us
    a durable queue: if a worker dies mid-task, the task isn't just lost.

Why Celery specifically over RQ: Celery has better support for retries with
backoff and task routing/priority queues, both of which this project will
plausibly want once ingestion has real production traffic (retry a failed
embedding call; prioritize small files over a 500-page PDF). RQ is simpler
to operate, which is a real advantage for a smaller project — the tradeoff
is Celery's extra operational surface (see docker-compose's `worker`
service) in exchange for headroom we're likely to use later in this
project specifically.
"""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "omnirag",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    # Belt-and-suspenders alongside DocumentStatus in Postgres: if a worker
    # is killed mid-task (OOM, deploy restart), the task is redelivered to
    # another worker instead of silently vanishing.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery_app.autodiscover_tasks(["app.workers"])
