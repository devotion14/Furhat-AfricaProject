"""Non-verbal + verbal cue primitives and the invitation routine.

An invitation is expressed as a sequence of cues whose parameters come from a
behaviour profile (config/profiles.json). This lets the *same* code produce
C1, C2, C3 and every Phase-1 co-design clip; only data changes.

Sequence (steps with 0 duration / null are skipped):
    1. group glance      look at the group centre          (group_glance_s)
    2. pair glance       look between invitee & neighbour  (pair_glance_s)
    3. orient            turn head/gaze to invitee         (orient_fraction)
    4. smile / nod       facial acknowledgement            (smile, nod)
    5. verbal prompt     English, or Kiswahili + English   (prompt_mode)
    6. hold gaze         keep looking at the invitee       (gaze_hold_s)
    7. wait              gaze at group or invitee           (gaze_during_wait, wait_s)
    8. release           if no answer: gentle "no problem" (release)  -> never insists
Steps 6-8 end early as soon as the invitee starts speaking.
"""

from __future__ import annotations

import math
import random
import threading
from typing import Callable

# Custom gestures (Furhat gesture JSON). Built-ins: Nod, Smile, BigSmile, ...
CUSTOM_GESTURES = {
    # A small, slow nod with a closed smile: acknowledging rather than demanding.
    "NuruSoftNod": {
        "name": "NuruSoftNod",
        "class": "furhatos.gestures.Gesture",
        "frames": [
            {"time": [0.35], "persist": False, "params": {"NECK_TILT": 6.0, "SMILE_CLOSED": 0.5}},
            {"time": [0.8], "persist": False, "params": {"NECK_TILT": 0.0, "SMILE_CLOSED": 0.4}},
            {"time": [1.3], "persist": False, "params": {"reset": True}},
        ],
    },
    # A warm closed-mouth smile held for ~2 s.
    "NuruWarmSmile": {
        "name": "NuruWarmSmile",
        "class": "furhatos.gestures.Gesture",
        "frames": [
            {"time": [0.4, 2.0], "persist": False, "params": {"SMILE_CLOSED": 0.6, "BROW_UP_LEFT": 0.2, "BROW_UP_RIGHT": 0.2}},
            {"time": [2.5], "persist": False, "params": {"reset": True}},
        ],
    },
}


def resolve_gesture(name: str | None):
    if not name:
        return None
    return CUSTOM_GESTURES.get(name, name)


def lerp(a, b, f: float):
    return tuple(x + f * (y - x) for x, y in zip(a, b))


def nearest_neighbour(seat: str, seats: dict) -> str | None:
    p = seats[seat]
    others = [s for s in seats if s != seat]
    if not others:
        return None
    return min(others, key=lambda s: math.dist(p, seats[s]))


def pick_prompt(profile: dict, rng: random.Random) -> list[tuple[str, str]]:
    """Return [(lang, text), ...] to speak, according to prompt_mode."""
    en = profile.get("prompts", {}).get("en") or []
    sw = profile.get("prompts", {}).get("sw") or []
    mode = profile.get("prompt_mode", "en")
    # Use the same index for both languages so the translation matches.
    i = rng.randrange(max(len(en), len(sw), 1))
    out = []
    if mode in ("bilingual", "sw") and sw:
        out.append(("sw", sw[i % len(sw)]))
    if mode in ("bilingual", "en") and en:
        out.append(("en", en[i % len(en)]))
    return out


def pick_release(profile: dict, rng: random.Random) -> list[tuple[str, str]]:
    rel = profile.get("release") or {}
    out = []
    for lang in ("sw", "en"):
        lines = rel.get(lang) or []
        if lines:
            out.append((lang, rng.choice(lines)))
    return out


def perform_invitation(robot, profile: dict, seat: str, seats: dict, centre,
                       responded: threading.Event,
                       on_prompt_end: Callable[[], None] = lambda: None,
                       emit: Callable[..., None] = lambda *a, **k: None,
                       rng: random.Random | None = None) -> bool:
    """Run the invitation cue sequence. Returns True if the invitee responded."""
    rng = rng or random.Random()

    def pause(s):  # interruptible wait; True if the invitee started speaking
        return responded.wait(s) if s > 0 else responded.is_set()

    if profile.get("group_glance_s", 0) > 0:
        robot.attend_location(centre)
        emit("cue", seat, "group_glance")
        pause(profile["group_glance_s"])

    if profile.get("pair_glance_s", 0) > 0:
        nb = nearest_neighbour(seat, seats)
        if nb:
            robot.attend_location(lerp(seats[seat], seats[nb], 0.5))
            emit("cue", seat, f"pair_glance:{nb}")
            pause(profile["pair_glance_s"])

    robot.attend_location(lerp(centre, seats[seat], float(profile.get("orient_fraction", 1.0))))
    emit("cue", seat, "orient")

    if profile.get("smile"):
        robot.gesture(CUSTOM_GESTURES["NuruWarmSmile"])
    nod = resolve_gesture(profile.get("nod"))
    if nod:
        robot.gesture(nod, blocking=True)
        emit("cue", seat, "nod")

    for lang, text in pick_prompt(profile, rng):
        if responded.is_set():
            break
        emit("robot_say", seat, text, lang=lang)     # logged at speech onset
        robot.say(text, lang=lang, blocking=True)

    on_prompt_end()
    emit("prompt_end", seat)

    if pause(float(profile.get("gaze_hold_s", 0))):
        return True
    if profile.get("gaze_during_wait") == "group":
        robot.attend_location(centre)
        emit("cue", seat, "return_to_group")
    if pause(float(profile.get("wait_s", 0))):
        return True

    # No answer: release the learner. The robot never repeats or insists.
    for lang, text in pick_release(profile, rng):
        emit("robot_say", seat, text, lang=lang)     # logged at speech onset
        robot.say(text, lang=lang, blocking=True)
    robot.attend_location(centre)
    emit("cue", seat, "release")
    return responded.is_set()
