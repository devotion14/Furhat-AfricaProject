"""Thin, fault-tolerant client for the Furhat Remote API (port 54321).

Why not use the furhat-remote-api package directly? It works, but it has no
request timeouts and raises on network hiccups. In a live study a crashed
script means a lost session, so this client:
  * uses explicit timeouts,
  * never raises on robot errors (it logs and returns False),
  * caches the active voice so bilingual prompts only switch when needed,
  * supports custom gestures (see nuru/cues.py) as well as built-in ones.

MockRobot has the same interface and prints instead of moving, so every
part of Nuru can be developed and tested without the robot.
"""

from __future__ import annotations

import json
import logging
import time

import requests

log = logging.getLogger("nuru.robot")


def fmt_point(p) -> str:
    return ",".join(f"{c:.3f}" for c in p)


class FurhatRobot:
    def __init__(self, host: str, port: int = 54321, timeout_s: float = 20.0,
                 voices: dict | None = None):
        self.base = f"http://{host}:{port}/furhat"
        self.timeout = timeout_s
        self.voices = voices or {}
        self._voice = None
        # Robot actions are serialised by the controller's worker thread; we
        # deliberately do not lock here so say/stop can interrupt from outside.
        self.session = requests.Session()

    # -- low level -----------------------------------------------------------
    def _call(self, method: str, path: str, params=None, body=None, timeout=None) -> dict | list | None:
        url = self.base + path
        try:
            r = self.session.request(method, url, params=params,
                                     data=json.dumps(body) if body is not None else None,
                                     headers={"Content-Type": "application/json"} if body is not None else None,
                                     timeout=timeout or self.timeout)
            r.raise_for_status()
            if not r.content:
                return {}
            try:
                return r.json()
            except ValueError:          # e.g. GET /furhat answers "Hello World!"
                return {"success": True, "message": r.text}
        except requests.RequestException as e:
            log.warning("Robot call %s %s failed: %s", method, path, e)
            return None

    def _ok(self, res) -> bool:
        return isinstance(res, dict) and res.get("success", False)

    # -- status --------------------------------------------------------------
    def ping(self) -> bool:
        return self._call("GET", "", timeout=3) is not None

    def users(self) -> list[dict]:
        res = self._call("GET", "/users", timeout=5)
        return res if isinstance(res, list) else []

    def voices_available(self) -> list[dict]:
        res = self._call("GET", "/voices")
        return res if isinstance(res, list) else []

    def gestures_available(self) -> list[str]:
        res = self._call("GET", "/gestures")
        return [g["name"] for g in res] if isinstance(res, list) else []

    # -- speech --------------------------------------------------------------
    def set_voice(self, name: str) -> bool:
        if name == self._voice:
            return True
        ok = self._ok(self._call("POST", "/voice", params={"name": name}))
        if ok:
            self._voice = name
        return ok

    def say(self, text: str, lang: str = "en", blocking: bool = True) -> bool:
        if not text:
            return True
        voice = self.voices.get(lang)
        if voice:
            self.set_voice(voice)
        return self._ok(self._call("POST", "/say", params={"text": text, "blocking": str(blocking).lower()},
                                   timeout=self.timeout + len(text) / 5))

    def stop_speaking(self) -> bool:
        return self._ok(self._call("POST", "/say/stop"))

    def listen(self, lang: str = "en-US") -> str:
        res = self._call("GET", "/listen", params={"language": lang}, timeout=30)
        return res.get("message", "") if isinstance(res, dict) else ""

    # -- gaze / head ---------------------------------------------------------
    def attend_location(self, point) -> bool:
        return self._ok(self._call("POST", "/attend", params={"location": fmt_point(point)}))

    def attend_user(self, user_id: str) -> bool:
        return self._ok(self._call("POST", "/attend", params={"userid": user_id}))

    # -- face ----------------------------------------------------------------
    def gesture(self, gesture, blocking: bool = False) -> bool:
        """gesture: name of a built-in gesture (str) or a definition (dict)."""
        if isinstance(gesture, dict):
            return self._ok(self._call("POST", "/gesture", params={"blocking": str(blocking).lower()}, body=gesture))
        return self._ok(self._call("POST", "/gesture", params={"name": gesture, "blocking": str(blocking).lower()}))

    def led(self, r: int = 0, g: int = 0, b: int = 0) -> bool:
        return self._ok(self._call("POST", "/led", params={"red": r, "green": g, "blue": b}))


class MockRobot:
    """Drop-in replacement that prints actions. speed < 1 makes speech faster."""

    def __init__(self, speed: float = 1.0, quiet: bool = False, **_):
        self.speed = speed
        self.quiet = quiet
        self.actions: list[tuple] = []

    def _do(self, *action):
        self.actions.append((time.time(), *action))
        if not self.quiet:
            print("  [robot]", *action)
        return True

    def ping(self): return True
    def users(self): return []
    def voices_available(self): return []
    def gestures_available(self): return ["Nod", "Smile", "BigSmile"]
    def set_voice(self, name): return self._do("voice", name)

    def say(self, text, lang="en", blocking=True):
        self._do("say", lang, text)
        if blocking:
            time.sleep(len(text) * 0.06 * self.speed)   # ~ speaking rate
        return True

    def stop_speaking(self): return self._do("say_stop")
    def listen(self, lang="en-US"): return ""
    def attend_location(self, point): return self._do("attend", fmt_point(point))
    def attend_user(self, user_id): return self._do("attend_user", user_id)

    def gesture(self, gesture, blocking=False):
        name = gesture["name"] if isinstance(gesture, dict) else gesture
        return self._do("gesture", name)

    def led(self, r=0, g=0, b=0): return self._do("led", r, g, b)


def make_robot(cfg, mock: bool = False, speed: float = 1.0):
    if mock:
        return MockRobot(speed=speed)
    return FurhatRobot(cfg.host, cfg.port, cfg.timeout_s, cfg.voices)
