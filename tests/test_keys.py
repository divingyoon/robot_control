"""Non-blocking single keys for a servo loop, and nothing when there is no tty."""

import io

from robot_control.keys import KeyReader


class FakeTty(io.StringIO):
    def __init__(self, text=""):
        super().__init__(text)
        self._pending = list(text)

    def isatty(self):
        return True

    def fileno(self):
        return 0


def test_reader_yields_nothing_without_a_tty():
    with KeyReader(stream=io.StringIO("q")) as keys:
        assert keys.poll() is None
        assert keys.enabled is False


def test_reader_polls_one_key_at_a_time(monkeypatch):
    pending = ["s", " "]
    reader = KeyReader(stream=FakeTty())
    monkeypatch.setattr(reader, "_enter_raw", lambda: None)
    monkeypatch.setattr(reader, "_leave_raw", lambda: None)
    monkeypatch.setattr(reader, "_read_ready", lambda: pending.pop(0) if pending else None)
    with reader as keys:
        assert keys.enabled is True
        assert keys.poll() == "s"
        assert keys.poll() == " "
        assert keys.poll() is None
