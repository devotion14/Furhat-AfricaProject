"""Per-participant voice-activity detection (VAD) from close-talk microphones.

Hardware: one lapel/headset mic per seat, all connected to ONE multichannel
audio interface (e.g. a 4-input USB interface). Channel i -> seat i.

Algorithm (per 30 ms frame, per channel)
    1. energy in dB (RMS)
    2. adaptive noise floor: slow-moving minimum tracker per channel
    3. a channel is "active" if energy > floor + threshold_db
    4. crosstalk suppression: close-talk mics also pick up neighbours, so only
       the LOUDEST active channel (by margin over its floor) is the speaker,
       and others only if within `crosstalk_db` of it (overlapping speech)
    5. hysteresis: speech must last `min_speech_s` to start a turn, and a
       turn ends only after `hangover_s` of silence (bridges short pauses)

Output: the same speak(seat) / silence() calls the Wizard-of-Oz panel makes,
so the participation model cannot tell the two input sources apart.
The wizard can always override from the panel (WoZ fallback).
"""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger("nuru.vad")


class MultiChannelVAD:
    def __init__(self, seats: list[str], frame_s: float = 0.03, threshold_db: float = 12.0,
                 crosstalk_db: float = 6.0, min_speech_s: float = 0.25, hangover_s: float = 0.8,
                 floor_rise_db_per_s: float = 0.5, on_start=None, on_stop=None):
        self.seats = seats
        self.frame_s = frame_s
        self.threshold_db = threshold_db
        self.crosstalk_db = crosstalk_db
        self.min_frames = max(1, round(min_speech_s / frame_s))
        self.hang_frames = max(1, round(hangover_s / frame_s))
        self.floor_rise = floor_rise_db_per_s * frame_s
        self.on_start = on_start or (lambda seat: None)
        self.on_stop = on_stop or (lambda seat: None)
        n = len(seats)
        self.floor = np.full(n, np.nan)
        self.run = np.zeros(n, dtype=int)       # consecutive active frames
        self.quiet = np.zeros(n, dtype=int)     # consecutive inactive frames while speaking
        self.speaking = np.zeros(n, dtype=bool)
        self.level_db = np.full(n, -120.0)

    @staticmethod
    def frame_db(frame: np.ndarray) -> np.ndarray:
        """frame: (samples, channels) float in [-1, 1] -> dB per channel."""
        rms = np.sqrt(np.mean(np.square(frame.astype(np.float64)), axis=0)) + 1e-10
        return 20 * np.log10(rms)

    def process(self, frame: np.ndarray) -> list[str]:
        """Feed one frame; returns list of seats currently speaking."""
        db = self.frame_db(frame)
        self.level_db = db
        # noise floor: drop instantly to quieter levels, rise slowly
        first = np.isnan(self.floor)
        self.floor[first] = db[first]
        self.floor = np.where(db < self.floor, db, self.floor + self.floor_rise)

        margin = db - self.floor
        active = margin > self.threshold_db
        if active.any():
            best = margin[active].max()
            active &= margin >= best - self.crosstalk_db

        for i, seat in enumerate(self.seats):
            if active[i]:
                self.run[i] += 1
                self.quiet[i] = 0
                if not self.speaking[i] and self.run[i] >= self.min_frames:
                    self.speaking[i] = True
                    self.on_start(seat)
            else:
                self.run[i] = 0
                if self.speaking[i]:
                    self.quiet[i] += 1
                    if self.quiet[i] >= self.hang_frames:
                        self.speaking[i] = False
                        self.quiet[i] = 0
                        self.on_stop(seat)
        return [s for s, on in zip(self.seats, self.speaking) if on]


class VADRunner:
    """Connects a live sounddevice input stream to a NuruController.

    The controller models ONE current speaker (as the study measures turns).
    With overlapping speech, the floor goes to whoever started first; when
    they stop, the next still-active speaker takes over.
    """

    def __init__(self, controller, device=None, samplerate: int = 16000, channels: list[int] | None = None, **vad_kw):
        import sounddevice as sd  # optional dependency
        self.sd = sd
        self.ctrl = controller
        self.seats = list(controller.seats)
        self.channels = channels or list(range(len(self.seats)))
        self.device = device
        self.sr = samplerate
        self.active: list[str] = []
        self.lock = threading.Lock()
        self.vad = MultiChannelVAD(self.seats, on_start=self._start, on_stop=self._stop, **vad_kw)
        self.stream = None

    def _start(self, seat):
        with self.lock:
            self.active.append(seat)
            if len(self.active) == 1:
                self.ctrl.speak(seat)

    def _stop(self, seat):
        with self.lock:
            was_floor = self.active and self.active[0] == seat
            if seat in self.active:
                self.active.remove(seat)
            if was_floor:
                if self.active:
                    self.ctrl.speak(self.active[0])
                else:
                    self.ctrl.silence()

    def _callback(self, indata, frames, t, status):
        if status:
            log.debug("audio status: %s", status)
        self.vad.process(indata[:, self.channels])

    def start(self):
        n_in = max(self.channels) + 1
        block = int(self.sr * self.vad.frame_s)
        self.stream = self.sd.InputStream(device=self.device, channels=n_in, samplerate=self.sr,
                                          blocksize=block, dtype="float32", callback=self._callback)
        self.stream.start()
        log.info("VAD started on device %s, channels %s -> seats %s", self.device, self.channels, self.seats)

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def levels(self) -> dict:
        return {s: round(float(d), 1) for s, d in zip(self.seats, self.vad.level_db)}
