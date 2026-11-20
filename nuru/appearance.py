"""What people look like when they arrive: expression, clothing colour and type.

One snapshot from the Furhat camera is sent to a vision model (Groq, set in
live.json -> "vision") while the robot is greeting, so there is no extra wait.
The model must answer with JSON; anything else is ignored. It is told to
describe only clearly visible things and never to guess age, gender,
ethnicity or identity.

How the result is used (live.json -> "appearance"):
  * "mention": "greeting"  the robot addresses each person once after the group
                           greeting ("Hello to you in the blue hoodie.") and adapts
                           the line to their expression (reassurance for a serious face)
  * "mention": "never"     nothing is said; expression still sets the tone
The snapshot is not saved; only the short description goes into the log.
"""

from __future__ import annotations

import base64
import logging

from .llm import ChatClient, extract_json, load_key

log = logging.getLogger("nuru.appearance")

PROMPT = (
    "These people are in front of a robot, facing the camera. For each clearly visible person, from the "
    "LEFT of the image to the RIGHT, describe only what is clearly visible. Never guess age, gender, "
    "ethnicity, religion or identity. Reply with JSON only:\n"
    '{"people": [{"expression": "smiling|calm|serious|surprised|nervous", '
    '"top_colour": "one simple colour word", '
    '"clothing": "short noun phrase, e.g. hoodie, t-shirt, school uniform, shirt, jacket, sweater, dress"}]}'
)

EXPRESSIONS = {"smiling", "calm", "serious", "surprised", "nervous"}

LINES = {
    "smiling": {"en": "I can see you're smiling. I like that energy!", "sw": "Naona unatabasamu, vizuri sana!"},
    "calm": {"en": "You look calm and ready.", "sw": "Unaonekana mtulivu."},
    "serious": {"en": "You look focused. There are no wrong answers today, so take your time.",
                "sw": "Usijali, hakuna jibu baya leo."},
    "surprised": {"en": "You look a little surprised. Don't worry, I'm friendly!", "sw": "Usishangae, mimi ni rafiki."},
    "nervous": {"en": "No pressure at all. You can always pass.", "sw": "Usijali, hakuna haraka."},
}


def snapshot_jpeg(host: str) -> bytes | None:
    """One frame from the Furhat camera feed, as JPEG bytes."""
    try:
        import cv2
        from .avsd import FurhatCamera
        cam = FurhatCamera(host, timeout_s=3.0)
        try:
            ok, frame = cam.read()
        finally:
            cam.release()
        if not ok:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None
    except Exception as e:
        log.warning("snapshot failed: %s", e)
        return None


def clean(obj: dict | None, n_expected: int) -> list[dict] | None:
    """Validate the model's JSON: known expressions, short words only."""
    if not obj or not isinstance(obj.get("people"), list):
        return None
    out = []
    for p in obj["people"][:6]:
        if not isinstance(p, dict):
            continue
        expr = str(p.get("expression", "")).lower().strip()
        colour = " ".join(str(p.get("top_colour", "")).lower().split()[:2])
        clothing = " ".join(str(p.get("clothing", "")).lower().split()[:3])
        out.append({"expression": expr if expr in EXPRESSIONS else "calm",
                    "colour": colour if colour.isascii() and len(colour) < 20 else "",
                    "clothing": clothing if clothing.isascii() and len(clothing) < 30 else ""})
    return out if out and len(out) == n_expected else None


class Appearance:
    def __init__(self, cfg, L: dict):
        V = L.get("vision", {})
        self.host = cfg.host
        self.client = ChatClient("vision", V.get("base_url", "https://api.groq.com/openai/v1"),
                                 V.get("model", ""), load_key(V.get("provider", "groq")),
                                 timeout_s=float(V.get("timeout_s", 6.0)), max_tokens=300,
                                 json_mode=bool(V.get("json_mode", True)))
        self.last_error = None

    @property
    def available(self) -> bool:
        return self.client.available and bool(self.client.model)

    def describe(self, n_people: int, jpeg: bytes | None = None) -> list[dict] | None:
        """-> one dict per person, left to right in the image, or None."""
        if not self.available:
            self.last_error = "no vision key/model"
            return None
        jpeg = jpeg or snapshot_jpeg(self.host)
        if not jpeg:
            self.last_error = "no camera frame"
            return None
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT},
                                             {"type": "image_url", "image_url": {"url": url}}]}]
        res = clean(extract_json(self.client.chat(msgs, temperature=0.1)), n_people)
        if res is None:
            self.last_error = self.client.last_error or "unusable reply / wrong number of people"
        return res


def greeting_line(d: dict, lang: str = "en") -> str:
    """'Hello to you in the blue hoodie. You look focused...' (English) or a short Kiswahili line."""
    expr = LINES.get(d.get("expression"), LINES["calm"])
    if lang == "sw":
        return expr["sw"]
    what = " ".join(x for x in (d.get("colour"), d.get("clothing")) if x)
    hello = f"Hello to you in the {what}." if what else "Hello to you."
    return f"{hello} {expr['en']}"
