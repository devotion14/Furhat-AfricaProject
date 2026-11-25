"""Recognising people who leave and come back (within ONE session).

Furhat's face tracker gives a person a new id every time they re-enter the
view. To know that "user-5" is the same person as "user-1" ten minutes ago, we
compare FACE SIGNATURES computed on the laptop with OpenCV's small models:

    YuNet   finds faces in a camera frame          (~0.2 MB, downloaded once to config/)
    SFace   turns a face into a 128-number signature (~37 MB, downloaded once)

Privacy: signatures live only in memory for the running session and are never
written to disk or sent anywhere. When the program stops they are gone.

The matching step (`FaceMemory.assign`) is plain numpy, so it is unit-tested
without real faces.
"""

from __future__ import annotations

import logging
import threading
import urllib.request

import numpy as np

from .config import CONFIG_DIR

log = logging.getLogger("nuru.faces")

MODELS = {
    "yunet": ("face_detection_yunet_2023mar.onnx",
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
              "face_detection_yunet_2023mar.onnx"),
    "sface": ("face_recognition_sface_2021dec.onnx",
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
              "face_recognition_sface_2021dec.onnx"),
}
SAME_PERSON = 0.363          # OpenCV's recommended cosine threshold for SFace


def model_path(key: str):
    name, url = MODELS[key]
    path = CONFIG_DIR / name
    if not path.exists():
        print(f"Downloading {name} (one time)...")
        urllib.request.urlretrieve(url, path)
    return path


def cosine(a, b) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class FaceMemory:
    """Session memory of faces: identity -> signature, and which tracker id it has now."""

    def __init__(self, threshold: float = SAME_PERSON):
        self.threshold = threshold
        self.identities: dict[str, dict] = {}     # "P1" -> {"feat": vec, "pid": tracker id, "n": count}
        self._lock = threading.Lock()

    def identity_of(self, pid: str) -> str | None:
        return next((i for i, d in self.identities.items() if d["pid"] == pid), None)

    def assign(self, pid: str, feat, present: list[str]) -> tuple[str, str | None]:
        """Register the face of tracker id `pid`.
        -> ("returning", old tracker id) when it matches someone who is NOT present now,
           ("known", None) when this pid already has an identity,
           ("new", None) otherwise."""
        with self._lock:
            if self.identity_of(pid):
                self._learn(self.identity_of(pid), feat)
                return "known", None
            best, sim = None, -1.0
            for iid, d in self.identities.items():
                if d["pid"] in present:            # that person is here: cannot be this face
                    continue
                s = cosine(feat, d["feat"])
                if s > sim:
                    best, sim = iid, s
            if best is not None and sim >= self.threshold:
                old = self.identities[best]["pid"]
                self.identities[best]["pid"] = pid
                self._learn(best, feat)
                return "returning", old
            iid = f"P{len(self.identities) + 1}"
            self.identities[iid] = {"feat": np.asarray(feat, float).ravel(), "pid": pid, "n": 1}
            return "new", None

    def _learn(self, iid, feat):
        d = self.identities[iid]
        d["feat"] = (d["feat"] * d["n"] + np.asarray(feat, float).ravel()) / (d["n"] + 1)
        d["n"] = min(d["n"] + 1, 10)               # keep adapting a little (lighting, angle)

    def forget_all(self):
        with self._lock:
            self.identities.clear()


class FaceSigner:
    """Camera frame -> face signatures, left to right (YuNet + SFace)."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(str(model_path("yunet")), "", (320, 320), 0.8)
        self.rec = cv2.FaceRecognizerSF.create(str(model_path("sface")), "")

    def signatures(self, frame) -> list[tuple[float, np.ndarray]]:
        """-> [(x centre 0..1, signature)] sorted left to right."""
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame)
        out = []
        for f in (faces if faces is not None else []):
            if f[2] * f[3] < (0.04 * w) * (0.04 * h):     # tiny faces (far away / background)
                continue
            aligned = self.rec.alignCrop(frame, f)
            out.append(((f[0] + f[2] / 2) / w, self.rec.feature(aligned).ravel()))
        return sorted(out, key=lambda t: t[0])
