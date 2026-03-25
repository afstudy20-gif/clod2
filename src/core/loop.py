"""Loop mode: run a prompt repeatedly on an interval."""
import threading
from typing import Callable


class LoopRunner:
    """Runs a prompt at a fixed interval in a background thread."""

    def __init__(self):
        self._timer: threading.Timer | None = None
        self._running = False
        self._prompt: str = ""
        self._interval: float = 0
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def prompt(self) -> str:
        return self._prompt

    @property
    def interval(self) -> float:
        return self._interval

    def start(
        self,
        prompt: str,
        interval: float,
        run_fn: Callable[[str], None],
    ):
        """Start running prompt every `interval` seconds.
        `run_fn` is called with the prompt each iteration.
        """
        self.stop()
        self._prompt = prompt
        self._interval = interval
        self._running = True
        self._run_fn = run_fn
        self._schedule_next()

    def _schedule_next(self):
        if not self._running:
            return
        self._timer = threading.Timer(self._interval, self._execute)
        self._timer.daemon = True
        self._timer.start()

    def _execute(self):
        if not self._running:
            return
        try:
            self._run_fn(self._prompt)
        except Exception as e:
            pass  # Silently continue on error
        self._schedule_next()

    def stop(self):
        """Stop the loop."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None
