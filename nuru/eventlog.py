"""Event log: one CSV row per event, flushed immediately (safe if the PC crashes).

Columns
    t          Unix time (s)
    t_rel      seconds since the session started
    group      group id, e.g. G03
    condition  c1 / c2 / c3 (empty between conditions)
    task       T1 / T2 / T3
    event      see EVENTS below
    seat       A-D (if relevant)
    value      event-specific value (latency, text, metric JSON, ...)
"""

from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

FIELDS = ["t", "t_rel", "group", "condition", "task", "event", "seat", "value"]

EVENTS = {
    "session_start", "session_end", "participant_meta",
    "condition_start", "condition_end", "task_question", "task_intro",
    "speak_start", "speak_stop",
    "invite", "cue", "robot_say", "prompt_end", "response", "no_response",
    "gaze", "metrics", "note", "robot_error",
}


class EventLog:
    def __init__(self, log_dir: Path, group: str = "G00"):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.group = group
        self.path = log_dir / f"nuru_{group}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS)
        self._w.writeheader()
        self._t0 = time.time()
        self._lock = threading.Lock()
        self.condition = ""
        self.task = ""
        self.tail: list[dict] = []

    def write(self, event: str, seat: str = "", value="", **extra):
        if extra:
            value = json.dumps({"value": value, **extra}, ensure_ascii=False) if value != "" else json.dumps(extra, ensure_ascii=False)
        elif isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        else:
            value = str(value)
        now = time.time()
        row = {"t": f"{now:.3f}", "t_rel": f"{now - self._t0:.3f}", "group": self.group,
               "condition": self.condition, "task": self.task, "event": event,
               "seat": seat or "", "value": value}
        with self._lock:
            self._w.writerow(row)
            self._f.flush()
            self.tail.append(row)
            del self.tail[:-50]
        return row

    def close(self):
        with self._lock:
            if not self._f.closed:
                self._f.close()
