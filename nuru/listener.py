"""Passive listening: the robot follows the discussion WITHOUT taking the floor.

    Furhat audio feed (or laptop mic)
        -> energy VAD cuts the stream into utterances; while the robot itself is
           speaking the mic is ignored (no echo turns)
        -> per utterance, in background threads:
              voiceprint  -> VoiceBank -> voice id (V1, V2, ...)        on the laptop
              Groq Whisper -> text + language (en / sw / mixed)        cloud
        -> on_utterance(Utterance)

The listener never makes the robot speak. It only reports what was said, by
which voice, and whether someone is talking right now (`someone_speaking`,
`silence_s`), which the floor rules in discussion.py use to avoid interrupting.

If transcription fails, the utterance is still reported (text=None): the robot
still knows THAT someone spoke, for how long and with which voice, so speaking
time and invitations keep working.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .vad import MultiChannelVAD

log = logging.getLogger("nuru.listener")
SR = 16000
FRAME = 480          # 30 ms at 16 kHz


@dataclass
class Utterance:
    start: float
    end: float
    text: str | None = None          # None = transcription failed / unavailable
    lang: str | None = None          # 'en', 'sw', ...
    voice: str | None = None         # V1, V2, ... (voice bank)
    person: str | None = None        # camera person id, once linked
    how: str = ""                    # how the person was decided
    audio: np.ndarray | None = field(default=None, repr=False)

    @property
    def duration(self) -> float:
        return self.end - self.start


class Listener:
    def __init__(self, on_utterance, transcriber=None, embedder=None, bank=None,
                 threshold_db: float = 10.0, end_gap_s: float = 0.7, min_s: float = 0.5, max_s: float = 15.0,
                 robot_tail_s: float = 0.4, clock=time.time):
        self.clock = clock
        self.on_utterance = on_utterance
        self.transcriber, self.embedder, self.bank = transcriber, embedder, bank
        self.vad = MultiChannelVAD(["mic"], frame_s=FRAME / SR, threshold_db=threshold_db, hangover_s=end_gap_s)
        self.robot_vad = MultiChannelVAD(["robot"], frame_s=FRAME / SR, threshold_db=10.0, hangover_s=robot_tail_s)
        self.min_s, self.max_s, self.robot_tail_s, self.end_gap_s = min_s, max_s, robot_tail_s, end_gap_s
        self._pre = collections.deque(maxlen=10)          # 300 ms pre-roll
        self._buf: list[np.ndarray] = []
        self._t_start = None
        self._robot_flag_until = 0.0
        self.robot_speaking = False
        self.someone_speaking = False
        self.last_speech_end = clock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nuru-stt")
        self._lock = threading.Lock()
        self._source = None
        self.error = None

    # -- the robot's own voice -------------------------------------------------
    def set_robot_speaking(self, on: bool):
        """Called around robot.say() when there is no robot audio channel (laptop mic)."""
        self._robot_flag_until = float("inf") if on else self.clock() + self.robot_tail_s

    def _robot_active(self, robot_frame) -> bool:
        if robot_frame is not None:
            self.robot_vad.process(robot_frame)
            if self.robot_vad.speaking[0]:
                return True
        return self.clock() < self._robot_flag_until

    # -- audio in -----------------------------------------------------------------
    def feed(self, mic: np.ndarray, robot: np.ndarray | None = None):
        """One 30 ms frame (float32). Separate from the audio source for testing."""
        with self._lock:
            self.robot_speaking = self._robot_active(robot)
            self.vad.process(mic.reshape(-1, 1))
            active = bool(self.vad.speaking[0]) and not self.robot_speaking
            now = self.clock()
            if active:
                if self._t_start is None:
                    self._t_start = now - len(self._pre) * FRAME / SR
                    self._buf = list(self._pre)
                self._buf.append(mic.reshape(-1))
                self.someone_speaking = True
                if now - self._t_start >= self.max_s:          # very long turn: cut it
                    self._finish(now)
            else:
                self._pre.append(mic.reshape(-1))
                if self._t_start is not None:
                    self._finish(now)
                self.someone_speaking = False

    def _finish(self, now):
        audio = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
        start, self._t_start, self._buf = self._t_start, None, []
        self.last_speech_end = now
        if len(audio) / SR - self.end_gap_s >= self.min_s:     # ignore coughs/clicks
            self._pool.submit(self._process, Utterance(start, now, audio=audio))

    def _process(self, u: Utterance):
        try:
            if self.embedder is not None and self.bank is not None:
                u.voice, _ = self.bank.identify(self.embedder(u.audio))
            if self.transcriber is not None:
                r = self.transcriber.transcribe(u.audio)
                if r:
                    u.text, u.lang = r["text"] or None, r["lang"]
            u.audio = None                                   # do not keep raw audio around
            self.on_utterance(u)
        except Exception:
            log.exception("utterance processing failed")

    def silence_s(self) -> float:
        return 0.0 if self.someone_speaking else self.clock() - self.last_speech_end

    # -- sources ------------------------------------------------------------------
    def start_furhat(self, ip: str, mic_channel: int = 0, robot_channel: int = 1):
        from .avsd import FurhatAudio
        self._source = FurhatAudio(ip, self.feed, mic_channel=mic_channel, robot_channel=robot_channel)
        self._source.start()

    def start_mic(self, device=None):
        import sounddevice as sd

        def cb(indata, frames, t, status):
            self.feed(indata[:, 0].copy())
        self._source = sd.InputStream(device=device, channels=1, samplerate=SR, blocksize=FRAME,
                                      dtype="float32", callback=cb)
        self._source.start()

    def stop(self):
        try:
            if self._source is not None:
                self._source.stop()
        except Exception:
            pass
        self._pool.shutdown(wait=False, cancel_futures=True)


class MockListener:
    """Scripted listener for --mock rehearsals: say(text, lang, voice, seconds)."""

    def __init__(self, on_utterance, speed: float = 1.0):
        self.on_utterance = on_utterance
        self.speed = speed
        self.someone_speaking = False
        self.robot_speaking = False
        self.last_speech_end = time.time()

    def set_robot_speaking(self, on: bool):
        self.robot_speaking = on

    def say(self, text, lang="en", voice="V1", seconds=3.0, person=None):
        self.someone_speaking = True
        t0 = time.time()
        time.sleep(seconds * self.speed)
        self.someone_speaking = False
        self.last_speech_end = time.time()
        self.on_utterance(Utterance(t0, self.last_speech_end, text, lang, voice, person))

    def silence_s(self) -> float:
        return 0.0 if self.someone_speaking else time.time() - self.last_speech_end

    def stop(self):
        pass
