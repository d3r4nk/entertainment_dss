"""Request-local instrumentation, shared by synchronous and background callers."""
from contextvars import ContextVar
from time import perf_counter

EVENT = ContextVar("chat_event", default=None)
DELTA = ContextVar("chat_delta", default=None)
OUTCOME = ContextVar("chat_outcome", default=None)


def emit(stage, status, **data):
    callback = EVENT.get()
    if callback:
        callback(dict(stage=stage, status=status, **data))


class Progress:
    def __init__(self):
        self.stage = None
        self.started = perf_counter()

    def next(self, stage, *, completed_status="ok", **data):
        if self.stage:
            emit(self.stage, completed_status, duration_ms=(perf_counter() - self.started) * 1000, **data)
        self.stage = stage
        self.started = perf_counter()
        if stage:
            emit(stage, "running")
