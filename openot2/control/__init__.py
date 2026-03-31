"""Generic task-controller data models and persistence."""

from .models import RunEvent, RunStatus, RunStep, StepStatus, TaskRun
from .store import JsonRunStore

__all__ = [
    "RunStatus",
    "StepStatus",
    "RunStep",
    "TaskRun",
    "RunEvent",
    "JsonRunStore",
    "create_app",
]


def create_app(*args, **kwargs):
    """Lazy-import wrapper so ``fastapi`` is not required at package import time."""
    from .web import create_app as _create_app

    return _create_app(*args, **kwargs)

