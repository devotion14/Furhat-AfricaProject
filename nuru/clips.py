"""Phase 1 co-design stimuli: short Furhat clips that vary ONE cue at a time.

The abstract lists the dimensions to vary: gaze duration, gaze target
(individual vs group), head orientation, wait time, invitation style. Each
clip below changes exactly one of these from a conventional BASELINE, so
participants' rankings can be attributed to that cue. V11 is the full
candidate Nuru bundle (current C3).

Record the robot (phone on a tripod, or screen capture of the virtual
Furhat), then run e.g.:
    python -m nuru clips --list
    python -m nuru clips V03            # one clip
    python -m nuru clips all            # every clip, with pauses between
A clip log with exact start/end times is written to logs/ to help cutting.
"""

from __future__ import annotations

import copy
import threading
import time

from .cues import perform_invitation
from .eventlog import EventLog

BASELINE = {
    "group_glance_s": 0.0, "pair_glance_s": 0.0, "orient_fraction": 1.0,
    "nod": None, "smile": False, "prompt_mode": "en",
    "prompts": {"en": ["What do you think?"], "sw": []},
    "gaze_hold_s": 3.0, "gaze_during_wait": "individual", "wait_s": 2.0,
    "release": {"en": [], "sw": []},
}

CLIPS = {
    "V00": ("Baseline (conventional)", "-", {}),
    "V01": ("Short individual gaze (1.5 s)", "gaze duration", {"gaze_hold_s": 1.5}),
    "V02": ("Long individual gaze (5 s)", "gaze duration", {"gaze_hold_s": 5.0}),
    "V03": ("Group glance before individual", "gaze target", {"group_glance_s": 1.2}),
    "V04": ("Neighbouring pair before individual", "gaze target", {"pair_glance_s": 1.2}),
    "V05": ("Whole-group invitation only", "gaze target",
            {"orient_fraction": 0.0, "prompts": {"en": ["What do you all think?"], "sw": []}}),
    "V06": ("Partial head turn towards learner", "head orientation", {"orient_fraction": 0.5}),
    "V07": ("Long wait (5 s), gaze back to group", "wait time",
            {"wait_s": 5.0, "gaze_during_wait": "group"}),
    "V08": ("Bilingual prompt (Kiswahili + English)", "invitation style",
            {"prompt_mode": "bilingual", "prompts": {"en": ["What do you think?"], "sw": ["Je, una maoni?"]}}),
    "V09": ("Acknowledging soft nod + smile", "invitation style", {"nod": "NuruSoftNod", "smile": True}),
    "V10": ("Explicit permission to pass", "invitation style",
            {"prompts": {"en": ["Would you like to add something? It is okay to pass."], "sw": []}}),
    "V11": ("Full candidate Nuru bundle (C3)", "bundle", "c3"),
}


def clip_profile(cfg, clip_id: str) -> dict:
    title, dim, ov = CLIPS[clip_id]
    if isinstance(ov, str):
        return cfg.profile(ov)
    p = copy.deepcopy(BASELINE)
    p.update(copy.deepcopy(ov))
    return p


def list_clips():
    print("ID   Dimension          Description")
    for k, (title, dim, _) in CLIPS.items():
        print(f"{k}  {dim:<18} {title}")


def run_clips(cfg, robot, ids: list[str], target: str | None = None, gap_s: float = 4.0):
    seats = cfg.seats
    target = target or list(seats)[1 % len(seats)]
    elog = EventLog(cfg.log_dir, "CLIPS")
    for n, cid in enumerate(ids):
        title, dim, _ = CLIPS[cid]
        prof = clip_profile(cfg, cid)
        print(f"\n=== {cid}: {title}  [{dim}]  target seat {target}")
        robot.attend_location(cfg.group_centre)
        time.sleep(1.5)
        elog.condition = cid
        elog.write("note", target, f"clip_start {cid} {title}")
        never = threading.Event()   # nobody answers in a clip
        perform_invitation(robot, prof, target, seats, cfg.group_centre, never,
                           emit=lambda *a, **k: elog.write(*a, **k))
        robot.attend_location(cfg.group_centre)
        time.sleep(1.0)
        elog.write("note", target, f"clip_end {cid}")
        if n < len(ids) - 1:
            print(f"   ...next clip in {gap_s:.0f} s")
            time.sleep(gap_s)
    elog.close()
    print(f"\nClip timings saved to {elog.path}")
