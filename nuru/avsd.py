"""Audio-visual speaker detection (AVSD): who is talking, from ONE webcam + ONE mic.

No extra hardware. Idea:
    1. the microphone tells us THAT someone is speaking (same energy VAD as vad.py)
    2. the camera tells us WHO: the face whose mouth is moving most at that moment
    3. faces are mapped to seats by where they are in the image (calibrated once)

Output: the same speak(seat) / silence() calls as the Wizard-of-Oz panel and
vad.py, so the participation model does not care where the input comes from.
The wizard can still override from the panel.

Install (needs Python 3.10-3.12 for MediaPipe):
    pip install mediapipe opencv-python sounddevice numpy pyzmq

Usage (run from the project folder):
    python -m nuru.avsd devices                 list cameras/mics
    python -m nuru.avsd calibrate               everyone sits down, 5 s
    python -m nuru.avsd test                    live window + prints who speaks
    options: --seats A B C  --camera 0  --mic 3  --flip  --threshold 12  --motion 0.004
             --furhat 192.168.1.110   use the camera inside Furhat instead of a webcam
             --furhat-mic             also use Furhat's microphone (Audio Feed); the robot's
                                      own speech is ignored automatically
    python -m nuru.avsd probe-audio --furhat IP    check the audio feed format/channels

Place the webcam just below/next to the robot, facing the participants, so every
face is visible. Keep the room reasonably quiet: the mic decides WHEN, the
camera decides WHO.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from .vad import MultiChannelVAD

log = logging.getLogger("nuru.avsd")
PROJECT = Path(__file__).resolve().parent.parent      # the NuruNew folder
CALIB_FILE = PROJECT / "config" / "avsd.json"

# MediaPipe Face Mesh landmark ids
UPPER_LIP, LOWER_LIP, FOREHEAD, CHIN, NOSE = 13, 14, 10, 152, 1


# --------------------------------------------------------------------------
# Pure decision logic (no camera / mic) - unit-testable
# --------------------------------------------------------------------------

def face_features(landmarks) -> tuple[float, float]:
    """MediaPipe landmarks -> (face centre x in [0,1], mouth opening / face height)."""
    lm = landmarks
    face_h = abs(lm[CHIN].y - lm[FOREHEAD].y) + 1e-6
    return lm[NOSE].x, abs(lm[LOWER_LIP].y - lm[UPPER_LIP].y) / face_h


class SpeakerDecider:
    """Combines 'speech heard' with per-seat mouth motion to pick the speaker.

    window_frames : how many video frames of mouth motion to look at (~0.5 s at 30 fps)
    min_motion    : below this mouth-opening variation nobody counts as talking
    switch_frames : a new person must be the most active this many frames in a row
                    before the floor switches to them (stops flicker)
    lost_frames   : speech heard but no moving mouth for this long -> silence
                    (e.g. the robot itself is talking, or a door slams)
    """

    def __init__(self, seats, seat_x, window_frames=15, min_motion=0.004,
                 switch_frames=4, lost_frames=10):
        self.seats = list(seats)
        self.seat_x = dict(seat_x)
        self.hist = {s: collections.deque(maxlen=window_frames) for s in self.seats}
        self.min_motion = min_motion
        self.switch_frames = switch_frames
        self.lost_frames = lost_frames
        self.current: str | None = None
        self._cand: str | None = None
        self._cand_n = 0
        self._no_motion = 0
        self.last_motion = {s: 0.0 for s in self.seats}

    def assign(self, faces):
        """faces: [(x, openness)] -> {seat: openness}; nearest seat centre, closest face wins."""
        best = {}
        for x, o in faces:
            seat = min(self.seats, key=lambda s: abs(self.seat_x[s] - x))
            d = abs(self.seat_x[seat] - x)
            if seat not in best or d < best[seat][0]:
                best[seat] = (d, o)
        return {s: o for s, (d, o) in best.items()}

    def motion(self):
        return {s: float(np.std(h)) if len(h) >= 3 else 0.0 for s, h in self.hist.items()}

    def update(self, faces, audio_active: bool) -> str | None:
        seen = self.assign(faces)
        for s in self.seats:
            if s in seen:
                self.hist[s].append(seen[s])
            else:
                self.hist[s].clear()  # face lost: no evidence for that seat
        self.last_motion = m = self.motion()

        if not audio_active:
            self.current, self._cand, self._cand_n, self._no_motion = None, None, 0, 0
            return None

        best = max(m, key=m.get)
        if m[best] < self.min_motion:
            # sound but no visibly moving mouth
            self._no_motion += 1
            if self._no_motion >= self.lost_frames:
                self.current = None
            return self.current
        self._no_motion = 0

        if best == self.current:
            self._cand, self._cand_n = None, 0
        elif self.current is None:
            self.current, self._cand, self._cand_n = best, None, 0
        else:
            if best == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = best, 1
            if self._cand_n >= self.switch_frames:
                self.current, self._cand, self._cand_n = best, None, 0
        return self.current


# --------------------------------------------------------------------------
# Camera sources: a local webcam, or the camera inside Furhat
# --------------------------------------------------------------------------

class FurhatCamera:
    """Reads the Furhat camera feed (enable it in the robot's web interface:
    Settings -> Camera feed). The robot publishes frames with ZeroMQ on port 3000.

    Messages on the feed are JPEG images and JSON annotations; we decode
    every message as an image and skip what is not one, and always return
    the NEWEST frame (older ones are dropped so we never lag behind).

    Same read()/isOpened()/release() interface as cv2.VideoCapture.
    """

    def __init__(self, ip: str, port: int = 3000, timeout_s: float = 5.0):
        import zmq
        self.zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVHWM, 4)
        self.sock.connect(f"tcp://{ip}:{port}")
        self.timeout_ms = int(timeout_s * 1000)
        self.url = f"tcp://{ip}:{port}"
        self._open = True

    @staticmethod
    def decode(msg: bytes):
        import cv2
        if len(msg) < 100 or not (msg[:2] == b"\xff\xd8" or msg[:4] == b"\x89PNG"):
            return None  # annotation / not an image
        return cv2.imdecode(np.frombuffer(msg, np.uint8), cv2.IMREAD_COLOR)

    def read(self):
        zmq = self.zmq
        if not self.sock.poll(self.timeout_ms):
            log.warning("No frames from Furhat camera at %s - is 'Camera feed' enabled?", self.url)
            return False, None
        latest = None
        while True:  # drain everything waiting, keep the newest image
            try:
                msg = self.sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            img = self.decode(msg)
            if img is not None:
                latest = img
        if latest is None:
            return self.read() if self.sock.poll(self.timeout_ms) else (False, None)
        return True, latest

    def isOpened(self):
        return self._open

    def release(self):
        self._open = False
        self.sock.close(0)


class FurhatAudio:
    """Reads the Furhat audio feed (Settings -> External Feeds -> Audio Feed, port 3001).

    Assumed format: raw 16-bit PCM, 16 kHz, 2 channels interleaved, one channel
    the robot's MICROPHONE (users), the other the robot's OWN SPEECH. Which is
    which can be set with mic_channel / robot_channel - check with
    'python -m nuru.avsd probe-audio --furhat IP'.

    Calls on_frame(mic_frame, robot_frame) with 30 ms float32 frames.
    """

    def __init__(self, ip, on_frame, port=3001, channels=2, frame_samples=480,
                 mic_channel=0, robot_channel=1):
        import zmq
        self.zmq = zmq
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(f"tcp://{ip}:{port}")
        self.url = f"tcp://{ip}:{port}"
        self.on_frame = on_frame
        self.channels = channels
        self.frame = frame_samples
        self.mic_ch = mic_channel
        self.robot_ch = robot_channel
        self.buf = np.zeros((0, channels), np.float32)
        self._stop = threading.Event()
        self._thread = None

    def feed(self, msg: bytes):
        """Turn one raw message into 30 ms frames. Separate from the socket for testing."""
        n = len(msg) // (2 * self.channels) * (2 * self.channels)
        if n == 0:
            return
        pcm = np.frombuffer(msg[:n], dtype="<i2").astype(np.float32) / 32768.0
        self.buf = np.vstack([self.buf, pcm.reshape(-1, self.channels)])
        while len(self.buf) >= self.frame:
            chunk, self.buf = self.buf[:self.frame], self.buf[self.frame:]
            robot = chunk[:, self.robot_ch:self.robot_ch + 1] if self.robot_ch is not None else None
            self.on_frame(chunk[:, self.mic_ch:self.mic_ch + 1], robot)

    def _loop(self):
        warned, last_data = False, time.time()
        try:
            while not self._stop.is_set():
                if not self.sock.poll(200):
                    if not warned and time.time() - last_data > 3.0:
                        log.warning("No audio from %s for 3 s - is 'Audio Feed' enabled?", self.url)
                        warned = True
                    continue
                last_data = time.time()
                self.feed(self.sock.recv())
        except Exception:
            if not self._stop.is_set():
                raise
        finally:
            self.sock.close(0)  # the socket is only closed by the thread that uses it

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def probe_audio(ip, seconds=6.0, port=3001):
    """Print what the audio feed sends, to confirm the format and channel order."""
    import zmq
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.connect(f"tcp://{ip}:{port}")
    print(f"Listening to tcp://{ip}:{port} for {seconds:.0f} s.")
    print("Talk for 3 s, then make the robot speak (e.g. Testing -> Speaking in the web UI).")
    t_end, n, sizes = time.time() + seconds, 0, set()
    while time.time() < t_end:
        if not sock.poll(500):
            continue
        msg = sock.recv()
        n += 1
        sizes.add(len(msg))
        if len(msg) >= 4 and len(msg) % 4 == 0:
            pcm = np.frombuffer(msg, "<i2").astype(np.float32).reshape(-1, 2) / 32768.0
            db = 20 * np.log10(np.sqrt(np.mean(pcm ** 2, axis=0)) + 1e-10)
            if n % 10 == 0:
                print(f"  ch0 {db[0]:6.1f} dB   ch1 {db[1]:6.1f} dB")
    sock.close(0)
    if n == 0:
        print("Nothing received - is 'Audio Feed' enabled and the IP right?")
    else:
        print(f"{n} messages, sizes {sorted(sizes)[:5]} bytes.")
        print("The channel that rises when YOU talk is the mic (--mic-channel),")
        print("the one that rises when the ROBOT talks is --robot-channel.")


MODEL_FILE = PROJECT / "config" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


def make_face_detector(max_faces):
    """Returns detect(rgb_image) -> list of faces, each a list of landmarks with .x/.y.

    Works with old MediaPipe (mp.solutions.face_mesh) and new MediaPipe
    (Tasks FaceLandmarker; downloads the 4 MB model once to config/).
    """
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=max_faces, refine_landmarks=False,
                                               min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def detect(rgb):
            res = mesh.process(rgb)
            return [Face(f.landmark) for f in (res.multi_face_landmarks or [])]
        return detect

    from mediapipe.tasks.python import BaseOptions, vision
    if not MODEL_FILE.exists():
        import urllib.request
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face model to {MODEL_FILE} (one time, ~4 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL_FILE)),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=max_faces,
                                        output_face_blendshapes=True)
    landmarker = vision.FaceLandmarker.create_from_options(opts)
    t0 = time.time()
    last = [-1]

    def detect(rgb):
        ts = max(int((time.time() - t0) * 1000), last[0] + 1)  # must strictly increase
        last[0] = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = landmarker.detect_for_video(img, ts)
        blends = getattr(res, "face_blendshapes", None) or []
        out = []
        for i, lm in enumerate(res.face_landmarks):
            b = {c.category_name: float(c.score) for c in blends[i]} if i < len(blends) else None
            out.append(Face(lm, b))
        return out
    detect.close = landmarker.close
    return detect


def open_camera(camera=0, furhat_ip=None):
    if furhat_ip:
        log.info("Using Furhat camera at %s:3000", furhat_ip)
        return FurhatCamera(furhat_ip)
    import cv2
    return cv2.VideoCapture(camera)


# --------------------------------------------------------------------------
# Social features: facial expression and clothing colour
# --------------------------------------------------------------------------

class Face(list):
    """A face's landmarks (list, with .x/.y) plus MediaPipe blendshapes if available."""

    def __init__(self, landmarks, blend=None):
        super().__init__(landmarks)
        self.blend = blend


MOUTH_L, MOUTH_R, CHEEK_L, CHEEK_R = 61, 291, 234, 454


def expression_from_blend(b: dict) -> str:
    """MediaPipe blendshape scores (0..1) -> smiling / surprised / serious / calm."""
    g = lambda *k: sum(b.get(x, 0.0) for x in k) / len(k)
    if g("mouthSmileLeft", "mouthSmileRight") > 0.4:
        return "smiling"
    if g("browInnerUp") > 0.5 and (g("jawOpen") > 0.25 or g("eyeWideLeft", "eyeWideRight") > 0.4):
        return "surprised"
    if g("browDownLeft", "browDownRight") > 0.35 or g("mouthFrownLeft", "mouthFrownRight") > 0.3:
        return "serious"
    return "calm"


def expression_from_landmarks(lm) -> str:
    """Fallback for old MediaPipe (no blendshapes): smile = wide mouth with lifted corners."""
    face_w = abs(lm[CHEEK_R].x - lm[CHEEK_L].x) + 1e-6
    face_h = abs(lm[CHIN].y - lm[FOREHEAD].y) + 1e-6
    mouth_w = abs(lm[MOUTH_R].x - lm[MOUTH_L].x) / face_w
    lift = ((lm[UPPER_LIP].y + lm[LOWER_LIP].y) / 2 - (lm[MOUTH_L].y + lm[MOUTH_R].y) / 2) / face_h
    return "smiling" if mouth_w > 0.44 or lift > 0.015 else "calm"


def face_expression(face) -> str:
    blend = getattr(face, "blend", None)
    return expression_from_blend(blend) if blend else expression_from_landmarks(face)


COLOUR_HUES = [(10, "red"), (22, "orange"), (34, "yellow"), (85, "green"),
               (130, "blue"), (150, "purple"), (172, "pink"), (180, "red")]   # OpenCV hue 0..180


def colour_name(h: float, s: float, v: float) -> str:
    """HSV (OpenCV ranges: h 0-180, s/v 0-255) -> a plain colour word."""
    if v < 60:
        return "black"
    if s < 45:
        return "white" if v > 185 else "grey"
    if 8 <= h <= 25 and v < 150:
        return "brown"
    for top, name in COLOUR_HUES:
        if h < top:
            return name
    return "red"


def clothing_colour(frame_bgr, face) -> str | None:
    """Main colour of the area just below the chin (the person's top)."""
    import cv2
    h, w = frame_bgr.shape[:2]
    xs = [face[CHEEK_L].x, face[CHEEK_R].x]
    face_h = abs(face[CHIN].y - face[FOREHEAD].y)
    x0, x1 = int(min(xs) * w), int(max(xs) * w)
    y0 = int((face[CHIN].y + 0.35 * face_h) * h)
    y1 = int((face[CHIN].y + 1.2 * face_h) * h)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None          # body not visible (face at the bottom of the image)
    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    sat = hsv[:, 1] >= 45
    if sat.mean() > 0.4:     # mostly coloured: use the hue of the coloured pixels
        hh = hsv[sat, 0]
        # circular median-ish: shift reds near 180 down so they group with reds near 0
        hh = np.where(hh > 170, hh - 180, hh)
        h_med = float(np.median(hh)) % 180
        return colour_name(h_med, float(np.median(hsv[sat, 1])), float(np.median(hsv[sat, 2])))
    return colour_name(0, float(np.median(hsv[:, 1])), float(np.median(hsv[:, 2])))


GREETING_LINES = {
    "smiling": "I can see you're smiling. I like that energy!",
    "calm": "You look calm and relaxed.",
    "serious": "You look focused. There are no wrong answers today, so take your time.",
    "surprised": "You look a little surprised. Don't worry, I'm friendly!",
}


def describe_sentence(obs: dict | None) -> str:
    """What the robot says to one participant, tailored to what it sees."""
    if not obs or not obs.get("expression"):
        return "Welcome! I can't see you very well yet, but I'm glad you're here."
    colour = obs.get("colour")
    hello = f"Hello to you in the {colour} top." if colour else "Hello to you."
    return f"{hello} {GREETING_LINES.get(obs['expression'], GREETING_LINES['calm'])}"


# --------------------------------------------------------------------------
# Calibration file
# --------------------------------------------------------------------------

def default_seat_x(seats):
    n = len(seats)
    return {s: (i + 0.5) / n for i, s in enumerate(seats)}


def load_seat_x(seats):
    try:
        data = json.loads(CALIB_FILE.read_text())
        sx = data["seat_x"]
        if all(s in sx for s in seats):
            return {s: float(sx[s]) for s in seats}
        log.warning("avsd.json does not match seats %s - using defaults", seats)
    except FileNotFoundError:
        log.warning("No %s yet - run 'python -m nuru.avsd calibrate'. Using defaults.", CALIB_FILE)
    return default_seat_x(seats)


def occupied_seats(seat_seen_t: dict, seat_x: dict, within_s: float = 2.0, now: float | None = None) -> list:
    """Calibrated seats where a face was seen in the last `within_s` seconds, left to right."""
    now = time.time() if now is None else now
    return sorted((s for s, t in seat_seen_t.items() if now - t <= within_s), key=lambda s: seat_x[s])


def calibrated_seats(default):
    """Seat names in config/avsd.json (any number), else the given default."""
    try:
        return list(json.loads(CALIB_FILE.read_text())["seat_x"])
    except (FileNotFoundError, KeyError, ValueError):
        return list(default)


def save_seat_x(seat_x):
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.write_text(json.dumps({"seat_x": seat_x}, indent=2))


# --------------------------------------------------------------------------
# Live runner (camera + mic)
# --------------------------------------------------------------------------

class AVSDRunner:
    """Same interface as vad.VADRunner: AVSDRunner(controller).start() / .stop().

    controller : object with .seats, .speak(seat), .silence()  (NuruController)
                 or None, then pass seats=[...] and on_change=print-like callback
    mute(True) : ignore the mic, e.g. while the robot itself is speaking
    """

    def __init__(self, controller=None, seats=None, camera=0, mic_device=None,
                 threshold_db=12.0, min_motion=0.004, show=False, on_change=None,
                 furhat_ip=None, furhat_audio=False, mic_channel=0, robot_channel=1):
        self.ctrl = controller
        self.seats = list(seats or controller.seats)
        self.camera = camera
        self.furhat_ip = furhat_ip
        self.furhat_audio = furhat_audio
        self.mic_channel = mic_channel
        self.robot_channel = robot_channel
        self.robot_speaking = False
        self._faudio = None
        # the robot's own voice: when it is speaking, ignore the mic (no echo turns)
        self.robot_vad = MultiChannelVAD(["robot"], threshold_db=10.0, hangover_s=0.5)
        self.mic_device = mic_device
        self.show = show
        self.on_change = on_change
        self.decider = SpeakerDecider(self.seats, load_seat_x(self.seats), min_motion=min_motion)
        self.audio_active = False
        self._muted = False
        self.audio_db = -120.0
        self._last = None
        self.error = None
        self.warning = None              # set while restarting after a failure
        self.max_restarts = 2
        self.n_faces = 0                 # faces in the latest frame
        self._present_since = None       # when someone appeared (None = nobody)
        self._last_face_t = 0.0
        self._obs = {s: {"expression": collections.deque(maxlen=30),
                         "colour": collections.deque(maxlen=30)} for s in self.seats}
        self._obs_lock = threading.Lock()
        self._frame_i = 0
        self.seat_seen_t: dict[str, float] = {}    # last time a face was seen in each seat
        self._stop = threading.Event()
        self._thread = None
        self._stream = None
        self.mic_vad = MultiChannelVAD(["mic"], threshold_db=threshold_db)

    # audio -------------------------------------------------------------
    def mute(self, muted: bool):
        """Ignore the mic, e.g. while the robot speaks (automatic with the Furhat audio feed)."""
        self._muted = muted

    def _process_audio(self, mic, robot=None):
        if robot is not None:
            self.robot_vad.process(robot)
            self.robot_speaking = bool(self.robot_vad.speaking[0])
        self.mic_vad.process(mic)
        self.audio_db = float(self.mic_vad.level_db[0])
        self.audio_active = (bool(self.mic_vad.speaking[0])
                             and not self._muted and not self.robot_speaking)

    def _audio_cb(self, indata, frames, t, status):
        self._process_audio(indata[:, :1])

    def _start_audio(self):
        if self.furhat_audio:
            self._faudio = FurhatAudio(self.furhat_ip, self._process_audio,
                                       mic_channel=self.mic_channel, robot_channel=self.robot_channel)
            self._faudio.start()
            return
        import sounddevice as sd
        block = int(16000 * self.mic_vad.frame_s)
        self._stream = sd.InputStream(device=self.mic_device, channels=1, samplerate=16000,
                                      blocksize=block, dtype="float32", callback=self._audio_cb)
        self._stream.start()

    # output ------------------------------------------------------------
    def _emit(self, seat):
        if seat == self._last:
            return
        self._last = seat
        if self.ctrl is not None:
            if seat:
                self.ctrl.speak(seat)
            else:
                self.ctrl.silence()
        if self.on_change:
            self.on_change(seat)

    # video loop --------------------------------------------------------
    def run_blocking(self):
        import cv2
        cap = open_camera(self.camera, self.furhat_ip)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera} - try --camera 1")
        detect = make_face_detector(len(self.seats))
        self._start_audio()
        log.info("AVSD running: camera %s, mic %s, seats %s", self.camera, self.mic_device, self.seats)
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                self.warning = None                 # frames are flowing again
                found = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self._update_presence(len(found))
                faces = [face_features(f) for f in found]
                self._observe(frame, found, faces)
                speaker = self.decider.update(faces, self.audio_active)
                self._emit(speaker)
                if self.show:
                    self._draw(cv2, frame, faces, speaker)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if hasattr(detect, "close"):
                try:
                    detect.close()          # release MediaPipe here, not at interpreter exit
                except Exception:
                    pass
            if self.show:
                cv2.destroyAllWindows()
            self._release_audio()

    # presence ----------------------------------------------------------
    def _update_presence(self, n):
        """Track how long someone has been in view (short dropouts < 0.5 s are ignored)."""
        now = time.time()
        self.n_faces = n
        if n:
            if self._present_since is None:
                self._present_since = now
            self._last_face_t = now
        elif now - self._last_face_t > 0.5:
            self._present_since = None

    def present_s(self) -> float:
        """Seconds that at least one face has been continuously in view (0 = nobody)."""
        return 0.0 if self._present_since is None else time.time() - self._present_since

    # social features ---------------------------------------------------
    def _observe(self, frame, found, faces):
        """Record expression (every frame) and clothing colour (every 5th frame) per seat."""
        self._frame_i += 1
        seen = {}
        for f, (x, _o) in zip(found, faces):
            seat = min(self.seats, key=lambda s: abs(self.decider.seat_x[s] - x))
            d = abs(self.decider.seat_x[seat] - x)
            if seat not in seen or d < seen[seat][0]:
                seen[seat] = (d, f)
        now = time.time()
        with self._obs_lock:
            for seat, (_d, f) in seen.items():
                self.seat_seen_t[seat] = now
                self._obs[seat]["expression"].append(face_expression(f))
                if self._frame_i % 5 == 0:
                    c = clothing_colour(frame, f)
                    if c:
                        self._obs[seat]["colour"].append(c)

    def observation(self, seat) -> dict | None:
        """Most frequent expression / colour over the last ~1 s / ~5 s for one seat."""
        with self._obs_lock:
            o = self._obs.get(seat)
            if not o or not o["expression"]:
                return None
            pick = lambda d: collections.Counter(d).most_common(1)[0][0] if d else None
            return {"expression": pick(o["expression"]), "colour": pick(o["colour"])}

    def _draw(self, cv2, frame, faces, speaker):
        h, w = frame.shape[:2]
        for s, x in self.decider.seat_x.items():
            px = int(x * w)
            col = (0, 220, 0) if s == speaker else (200, 200, 200)
            cv2.line(frame, (px, 0), (px, h), col, 1)
            mot = self.decider.last_motion.get(s, 0.0)
            cv2.putText(frame, f"{s}  {mot * 1000:4.1f}", (px - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            ob = self.observation(s)
            if ob:
                cv2.putText(frame, f"{ob['expression']}, {ob['colour'] or '?'} top", (px - 60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        for x, o in faces:
            cv2.circle(frame, (int(x * w), h // 2), 6, (255, 160, 0), -1)
        state = "ROBOT" if self.robot_speaking else ("SPEECH" if self.audio_active else "quiet")
        status = f"mic {self.audio_db:6.1f} dB  {state}  ->  {speaker or '-'}"
        cv2.putText(frame, status, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.imshow("Nuru AVSD (q to quit)", frame)

    def start(self):
        """Background mode for the panel / session (no window)."""
        self.show = False
        self.error = None

        def run():
            # Restart on failure (camera feed dropped, network hiccup) before giving up.
            # While down we report silence, so the controller never keeps a stale speaker.
            for attempt in range(self.max_restarts + 1):
                try:
                    self.run_blocking()
                    return                          # stopped on purpose
                except Exception as e:
                    if self._stop.is_set():
                        return                              # shutting down (Ctrl+C): not a failure
                    log.exception("AVSD failed (attempt %d)", attempt + 1)
                    self._emit(None)
                    self._present_since = None
                    if attempt < self.max_restarts:
                        self.warning = f"Speaker detection restarting after: {e}"
                        time.sleep(2.0)
                    else:   # report to the panel / session instead of dying silently
                        self.error = f"Speaker detection stopped: {e}"
            self.warning = None
        self._thread = threading.Thread(target=run, daemon=True, name="nuru-avsd")
        self._thread.start()

    def levels(self) -> dict:
        """For the panel's level display (same method name as vad.VADRunner)."""
        return {"mic": round(self.audio_db, 1), "robot_speaking": self.robot_speaking}

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)      # let the video loop finish its frame and close MediaPipe
        self._release_audio()

    def _release_audio(self):
        if self._faudio:
            self._faudio.stop()
            self._faudio = None
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def calibrate(seats, camera=0, seconds=5.0, flip=False, furhat_ip=None):
    """Everyone sits in their seat facing the camera. Seats are assigned left->right
    in the image (use --flip if A should be on the right of the image)."""
    import cv2

    cap = open_camera(camera, furhat_ip)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {camera} - try --camera 1")
    detect = make_face_detector(len(seats))
    samples = []
    print(f"Calibrating for {seconds:.0f} s: {len(seats)} people, sit still and face the camera...")
    t_end = time.time() + seconds
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue
        xs = sorted(face_features(f)[0] for f in detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if len(xs) == len(seats):
            samples.append(xs)
    cap.release()
    if len(samples) < 10:
        raise SystemExit(f"Saw all {len(seats)} faces in only {len(samples)} frames. "
                         "Move the camera so everyone is fully visible, then retry.")
    centres = np.mean(samples, axis=0)
    order = list(reversed(seats)) if flip else list(seats)
    seat_x = {s: round(float(x), 3) for s, x in zip(order, centres)}
    save_seat_x(seat_x)
    print(f"Saved {CALIB_FILE}: {seat_x}")
    print("Now run 'python -m nuru.avsd test' and check each label sits on the right person.")


def list_devices():
    import sounddevice as sd
    print("Microphones (use the number with --mic):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i:3d}  {d['name']}")
    try:
        import cv2
        print("Cameras (use the number with --camera):")
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"  {i:3d}  camera found")
            cap.release()
    except ImportError:
        print("opencv-python not installed")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m nuru.avsd", description="Camera + mic speaker detection")
    p.add_argument("command", choices=["devices", "calibrate", "test", "probe-audio"])
    p.add_argument("--seats", nargs="+", default=["A", "B", "C"])
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mic", type=int, default=None)
    p.add_argument("--furhat", metavar="IP", default=None,
                   help="use the camera inside Furhat (enable 'Camera feed' in the robot's Settings)")
    p.add_argument("--furhat-mic", action="store_true",
                   help="use the robot's microphone via the Audio Feed (needs --furhat IP)")
    p.add_argument("--mic-channel", type=int, default=0)
    p.add_argument("--robot-channel", type=int, default=1)
    p.add_argument("--flip", action="store_true", help="seat A on the RIGHT of the camera image")
    p.add_argument("--threshold", type=float, default=12.0, help="mic sensitivity in dB (lower = more sensitive)")
    p.add_argument("--motion", type=float, default=0.004, help="min mouth motion (lower = more sensitive)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if a.furhat_mic and not a.furhat:
        p.error("--furhat-mic needs --furhat IP")
    if a.command == "devices":
        list_devices()
    elif a.command == "probe-audio":
        if not a.furhat:
            p.error("probe-audio needs --furhat IP")
        probe_audio(a.furhat)
    elif a.command == "calibrate":
        calibrate(a.seats, camera=a.camera, flip=a.flip, furhat_ip=a.furhat)
    else:
        r = AVSDRunner(seats=a.seats, camera=a.camera, mic_device=a.mic, threshold_db=a.threshold,
                       min_motion=a.motion, show=True, furhat_ip=a.furhat,
                       furhat_audio=a.furhat_mic, mic_channel=a.mic_channel,
                       robot_channel=a.robot_channel,
                       on_change=lambda s: print(f"SPEAK {s}" if s else "silence"))
        r.run_blocking()


if __name__ == "__main__":
    main()
