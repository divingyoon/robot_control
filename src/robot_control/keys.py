"""Single keystrokes for a servo loop, read without blocking it.

The terminal is put in cbreak mode, not raw: keys arrive one at a time without
a newline, but Ctrl-C still raises KeyboardInterrupt, which is the loop's
stop. Without a tty (a test, a pipe, a launch file) nothing is ever read and
the loop runs on its timer alone.
"""

from __future__ import annotations

import select
import sys
from typing import Any


class KeyReader:
    def __init__(self, stream: Any | None = None):
        self._stream = sys.stdin if stream is None else stream
        self._saved: Any = None
        self.enabled = False

    def __enter__(self) -> KeyReader:
        try:
            self.enabled = bool(self._stream.isatty())
        except (AttributeError, ValueError, OSError):
            self.enabled = False
        if self.enabled:
            self._enter_raw()
        return self

    def __exit__(self, *_exception: object) -> None:
        if self.enabled:
            self._leave_raw()
            self.enabled = False

    def poll(self) -> str | None:
        """The next key if one is waiting, else None. Never blocks."""
        if not self.enabled:
            return None
        return self._read_ready()

    def _enter_raw(self) -> None:
        import termios
        import tty

        descriptor = self._stream.fileno()
        self._saved = termios.tcgetattr(descriptor)
        tty.setcbreak(descriptor)

    def _leave_raw(self) -> None:
        import termios

        if self._saved is not None:
            termios.tcsetattr(self._stream.fileno(), termios.TCSADRAIN, self._saved)
            self._saved = None

    def _read_ready(self) -> str | None:
        ready, _, _ = select.select([self._stream], [], [], 0.0)
        if not ready:
            return None
        key = self._stream.read(1)
        return key or None
