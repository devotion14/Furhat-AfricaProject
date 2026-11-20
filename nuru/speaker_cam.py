"""Camera hint for WHO is speaking (mouth movement), to link voices to faces.

Wraps devotion14's audio-visual speaker detector (nuru/avsd.py): the Furhat
camera feed + MediaPipe face landmarks tell which calibrated seat has a moving
mouth while the microphone hears speech. We keep a short timeline of those
decisions and, for each utterance, return the seat that was speaking for most
of it, mapped to the camera person (Furhat face id) in the same left-right
position.

Optional: needs mediapipe + opencv + pyzmq and 'Camera feed' / 'Audio feed'
enabled on the robot. If anything fails, `available` is False and the
discussion runs on voiceprints alone.
"""

from __future__ import annotations

import collections
import logging
import threading
import time

log = logging.getLogger("nuru.speaker_cam")


class CameraSpeakerHint:
    def __init__(self, cfg, D: dict):
        from .avsd import AVSDRunner
        self.seats = list(cfg.seats)
        self.flip = bool(D.get("camera_flip", False))
        self.events = collections.deque(maxlen=4000)      # (time, seat or None)
        self._lock = threading.Lock()
        self.det = AVSDRunner(seats=self.seats, camera=0, furhat_ip=cfg.host, furhat_audio=True,
                              threshold_db=float(D.get("camera_threshold_db", 9.0)), on_change=self._on)
        self.det.start()

    @property
    def available(self) -> bool:
        return not self.det.error

    def _on(self, seat):
        with self._lock:
            self.events.append((time.time(), seat))

    def seat_during(self, start: float, end: float) -> str | None:
        """Seat whose mouth moved for at least half of [start, end]."""
        with self._lock:
            ev = list(self.events)
        if end <= start:
            return None
        totals: dict = {}
        for i, (t, seat) in enumerate(ev):
            t_next = ev[i + 1][0] if i + 1 < len(ev) else end
            a, b = max(t, start), min(t_next, end)
            if seat and b > a:
                totals[seat] = totals.get(seat, 0.0) + (b - a)
        if not totals:
            return None
        seat, secs = max(totals.items(), key=lambda kv: kv[1])
        return seat if secs >= 0.5 * (end - start) else None

    def person_for(self, seat: str | None, people) -> str | None:
        """Map a camera seat to a Furhat face id by left-right order (needs equal counts)."""
        if not seat or len(people) != len(self.seats):
            return None
        order = sorted(self.seats, key=lambda s: self.det.decider.seat_x[s], reverse=self.flip)
        ppl = sorted(people, key=lambda x: x.location[0])
        return ppl[order.index(seat)].id

    def stop(self):
        self.det.stop()
