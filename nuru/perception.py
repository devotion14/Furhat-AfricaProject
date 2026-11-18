"""SENSE: what the robot perceives about the people in front of it.

Everything the live loop knows about people comes from here, and from nowhere
else. Two sources, both on the robot:

  * Furhat's camera + on-robot face tracker  ->  GET /furhat/users
        in : nothing (polled every `poll_s`)
        out: list of {id, location {x, y, z} in metres}
  * Furhat's microphone + cloud speech recogniser  ->  GET /furhat/listen
        (used in live.py, ACT/SENSE boundary; we only use WHETHER someone
         spoke and for how long, never WHAT they said)

This module turns the raw face list into a `Percept`: who is present, how far
away they are, how long each has been there, and who just arrived or left.
That is the full interface between perception and the decision step.

Robustness: a failed /users call never raises. A face must be missing for
`lost_after_s` before the person counts as gone (the tracker drops faces for a
few frames when people turn their heads), and must be seen for `confirm_s`
before counting as present (people walking past are ignored).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Person:
    id: str
    location: tuple[float, float, float]
    first_seen: float
    last_seen: float

    @property
    def distance(self) -> float:
        x, _, z = self.location
        return math.hypot(x, z)


@dataclass
class Percept:
    """A snapshot of the situation. This is the ONLY input to the plan step."""
    t: float
    people: list[Person] = field(default_factory=list)   # confirmed, left -> right
    arrived: list[str] = field(default_factory=list)     # ids confirmed since the last snapshot
    left: list[str] = field(default_factory=list)        # ids lost since the last snapshot
    camera_ok: bool = True

    @property
    def n(self) -> int:
        return len(self.people)

    def nearest(self) -> Person | None:
        return min(self.people, key=lambda p: p.distance, default=None)

    def dwell_s(self, pid: str) -> float:
        p = self.get(pid)
        return self.t - p.first_seen if p else 0.0

    def get(self, pid: str) -> Person | None:
        return next((p for p in self.people if p.id == pid), None)


class PeopleTracker:
    """Polls the robot's face tracker in a background thread."""

    def __init__(self, robot, poll_s: float = 0.3, confirm_s: float = 1.0, lost_after_s: float = 3.0,
                 max_distance_m: float = 2.5, clock: Callable[[], float] = time.time):
        self.robot = robot
        self.poll_s = poll_s
        self.confirm_s = confirm_s
        self.lost_after_s = lost_after_s
        self.max_distance_m = max_distance_m
        self.clock = clock
        self._seen: dict[str, Person] = {}        # every face currently tracked (confirmed or not)
        self._confirmed: set[str] = set()
        self._arrived: list[str] = []
        self._left: list[str] = []
        self._camera_ok = True
        self._fails = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- raw input -----------------------------------------------------------
    def update(self, users: list[dict] | None):
        """Feed one /users result (None = the call failed)."""
        now = self.clock()
        with self._lock:
            if users is None:
                self._fails += 1
                self._camera_ok = self._fails < 5          # ~1.5 s of failures
                return                                     # keep the last known state
            self._fails, self._camera_ok = 0, True
            for u in users:
                loc = u.get("location") or {}
                if not all(k in loc for k in ("x", "y", "z")):
                    continue
                pid = str(u.get("id", ""))
                point = (float(loc["x"]), float(loc["y"]), float(loc["z"]))
                if math.hypot(point[0], point[2]) > self.max_distance_m:
                    continue                               # too far away to be "in the interaction"
                p = self._seen.get(pid)
                if p:
                    p.location, p.last_seen = point, now
                else:
                    self._seen[pid] = Person(pid, point, now, now)
            for pid, p in list(self._seen.items()):
                if pid not in self._confirmed and now - p.first_seen >= self.confirm_s \
                        and now - p.last_seen < self.poll_s * 3:
                    self._confirmed.add(pid)
                    self._arrived.append(pid)
                if now - p.last_seen > self.lost_after_s:
                    del self._seen[pid]
                    if pid in self._confirmed:
                        self._confirmed.discard(pid)
                        self._left.append(pid)

    def snapshot(self) -> Percept:
        """Current percept; arrived/left are reported once each."""
        with self._lock:
            people = sorted((self._seen[i] for i in self._confirmed), key=lambda p: p.location[0])
            pc = Percept(self.clock(), [Person(p.id, p.location, p.first_seen, p.last_seen) for p in people],
                         self._arrived, self._left, self._camera_ok)
            self._arrived, self._left = [], []
            return pc

    # -- background polling --------------------------------------------------
    def _poll(self):
        while not self._stop.is_set():
            try:
                users = self.robot.users_or_none()
            except Exception:
                users = None
            self.update(users)
            self._stop.wait(self.poll_s)

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True, name="nuru-sense")
        self._thread.start()

    def stop(self):
        self._stop.set()
