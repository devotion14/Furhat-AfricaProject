"""NuruController: the moderation engine shared by the web panel, CLI and VAD.

Threads
    caller threads  (web panel / keyboard / VAD)  -> speak(), silence(), invite() ...
    worker thread   executes robot actions one at a time from a queue, so the
                    operator never waits for the robot and gaze never "fights"
    supervisor      every 0.2 s: closes unanswered invitations, and (if auto
                    moderation is on) invites after a silence

All tracker access is guarded by one lock.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time

from .cues import CUSTOM_GESTURES, lerp, perform_invitation, resolve_gesture
from .eventlog import EventLog
from .participation import Tracker

log = logging.getLogger("nuru.controller")


def auto_invite_rule(mod: dict, someone_speaking: bool, silence_s: float, since_invite_s: float) -> str | None:
    """Decide whether the robot should invite now. Returns the rule name or None.

    lull      : the group has been silent for auto_invite_silence_s
    turn_end  : a turn just ended (turn_end_gap_s of silence) and at least
                turn_end_invite_interval_s have passed since the last invitation
    """
    if someone_speaking or since_invite_s < float(mod.get("min_gap_between_invites_s", 12)):
        return None
    if silence_s >= float(mod.get("auto_invite_silence_s", 7)):
        return "lull"
    interval = float(mod.get("turn_end_invite_interval_s", 0) or 0)
    if interval > 0 and silence_s >= float(mod.get("turn_end_gap_s", 1.5)) and since_invite_s >= interval:
        return "turn_end"
    return None


def seats_from_users(users: list[dict], seats: dict) -> dict | None:
    """Assign faces detected by Furhat to seats, matching left-to-right order.

    Returns {seat: (x, y, z)} or None if the number of faces != number of seats.
    """
    pts = []
    for u in users:
        loc = u.get("location") or {}
        if all(k in loc for k in ("x", "y", "z")):
            pts.append((float(loc["x"]), float(loc["y"]), float(loc["z"])))
    ids = list(seats)
    if len(pts) != len(ids):
        return None
    ascending = seats[ids[0]][0] <= seats[ids[-1]][0]   # keep the configured x-ordering
    pts.sort(key=lambda p: p[0], reverse=not ascending)
    return dict(zip(ids, pts))


class NuruController:
    def __init__(self, cfg, robot, group: str = "G00", seed: int | None = None):
        self.cfg = cfg
        self.robot = robot
        self.seats = cfg.seats
        self.centre = cfg.group_centre
        self.rng = random.Random(seed)
        self.lock = threading.RLock()

        self.log = EventLog(cfg.log_dir, group)
        self.group = group
        self.condition: str | None = None
        self.profile: dict = cfg.profile("c1")
        self.task: str | None = None
        self.question_idx = -1
        self.auto = bool(cfg.mod("auto_invite", False))
        self.follow_speaker = bool(cfg.mod("follow_speaker", True))
        self.condition_started: float | None = None
        self.results: dict[str, dict] = {}
        self.meta: dict[str, str] = {}

        self.tracker = self._new_tracker()
        self._idle_idx = 0
        self._last_invite_t = 0.0
        self._inviting: str | None = None
        self._floor_taken = threading.Event()
        self._gaze_seq = 0
        self.status = "idle"

        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._work, daemon=True, name="nuru-robot")
        self._super = threading.Thread(target=self._supervise, daemon=True, name="nuru-supervisor")
        self._worker.start()
        self._super.start()
        self.log.write("session_start", value={"seats": list(self.seats), "robot": f"{cfg.host}:{cfg.port}"})

    def _new_tracker(self) -> Tracker:
        return Tracker(list(self.seats), no_repeat_window=int(self.cfg.mod("no_repeat_window", 2)),
                       response_window_s=float(self.cfg.mod("response_window_s", 8.0)))

    # ------------------------------------------------------------------ worker
    def _enqueue(self, fn, *args, label=""):
        self._q.put((fn, args, label))

    def _work(self):
        while not self._stop.is_set():
            try:
                fn, args, label = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if label:
                    self.status = label
                fn(*args)
            except Exception as e:  # never let the worker die mid-session
                log.exception("robot action failed")
                self.log.write("robot_error", value=str(e))
            finally:
                self.status = "idle" if self._q.empty() else self.status

    def _emit(self, event, seat="", value="", **extra):
        self.log.write(event, seat, value, **extra)

    # ------------------------------------------------------------- supervisor
    def _supervise(self):
        while not self._stop.is_set():
            time.sleep(0.2)
            with self.lock:
                if self.condition is None:
                    continue
                missed = self.tracker.expire_invitation()
                if missed:
                    self.log.write("no_response", missed.seat)
                if self.auto and self._inviting is None and self._q.empty():
                    rule = auto_invite_rule(self.cfg.moderation, self.tracker.current is not None,
                                            self.tracker.silence_s(), time.time() - self._last_invite_t)
                    if rule:
                        self.invite(source=f"auto:{rule}")

    # ------------------------------------------------------------- conditions
    def set_meta(self, meta: dict[str, str]):
        """Self-reported usual participation level per seat (quiet/average/active)."""
        self.meta = dict(meta)
        for seat, level in meta.items():
            self.log.write("participant_meta", seat, level)

    def start_condition(self, condition: str, task: str | None = None, intro: bool = True):
        condition = condition.lower()
        with self.lock:
            if self.condition:
                self.end_condition()
            self.condition = condition
            self.profile = self.cfg.profile(condition)
            self.tracker = self._new_tracker()
            self.task = task
            self.question_idx = -1
            self.condition_started = time.time()
            self.log.condition, self.log.task = condition, task or ""
            self.log.write("condition_start", value={"profile": self.profile.get("label"), "task": task})
        self._enqueue(self.robot.attend_location, self.centre, label="look at group")
        if intro and task:
            t = self.cfg.tasks["tasks"][task]
            self.log.write("task_intro", value=t["intro"])
            self._enqueue(self._say_group, t["intro"], "en", label="task intro")

    def end_condition(self) -> dict | None:
        with self.lock:
            if not self.condition:
                return None
            self._stop_current()
            m = self.tracker.metrics()
            m["duration_s"] = round(time.time() - (self.condition_started or time.time()), 1)
            self.results[self.condition] = m
            self.log.write("metrics", value=m)
            self.log.write("condition_end")
            self.condition = None
            self.log.condition, self.log.task = "", ""
            return m

    # ----------------------------------------------------------------- script
    def greet(self):
        g = self.cfg.tasks.get("greeting", {})
        if g.get("sw"):
            self._enqueue(self._say_group, g["sw"], "sw", label="greeting")
        if g.get("en"):
            self._enqueue(self._say_group, g["en"], "en", label="greeting")

    def close(self):
        c = self.cfg.tasks.get("closing", {})
        for lang in ("sw", "en"):
            if c.get(lang):
                self._enqueue(self._say_group, c[lang], lang, label="closing")

    def next_question(self) -> str | None:
        if not self.task:
            return None
        qs = self.cfg.tasks["tasks"][self.task]["questions"]
        if self.question_idx + 1 >= len(qs):
            return None
        self.question_idx += 1
        q = qs[self.question_idx]
        self.log.write("task_question", value=q, index=self.question_idx)
        with self.lock:
            self._stop_current()         # the robot takes the floor
            self.tracker.last_speech_end = time.time()
        self._enqueue(self._say_group, q, "en", label=f"question {self.question_idx + 1}")
        return q

    def _say_group(self, text, lang):
        self.robot.attend_location(self.centre)
        self.log.write("robot_say", value=text, lang=lang)
        self.robot.say(text, lang=lang, blocking=True)
        with self.lock:
            self.tracker.last_speech_end = time.time()   # silence counts from end of robot speech

    def say(self, text: str, lang: str = "en"):
        self._enqueue(self._say_free, text, lang, label="say")

    def _say_free(self, text, lang):
        self.log.write("robot_say", value=text, lang=lang, free=True)
        self.robot.say(text, lang=lang, blocking=True)

    # ----------------------------------------------------------------- speech
    def _stop_current(self):
        """End the current turn (if any) and log it. Caller holds the lock."""
        if self.tracker.current is not None:
            seat = self.tracker.current
            self.tracker.stop()
            self.log.write("speak_stop", seat)

    def speak(self, seat: str):
        seat = seat.upper()
        if seat not in self.seats:
            return
        with self.lock:
            if seat == self.tracker.current:
                return
            self._stop_current()
            info = self.tracker.start(seat)
            self.log.write("speak_start", seat)
            if info.get("response"):
                self.log.write("response", seat, round(info["latency_s"], 3))
            self._floor_taken.set()
            inviting = self._inviting is not None
        if self.follow_speaker and not inviting:
            self._gaze_to(seat)

    def silence(self):
        with self.lock:
            if self.tracker.current is None:
                return
            self._stop_current()
            inviting = self._inviting is not None
        if not inviting and self.condition:
            self._idle_gaze()

    def _gaze_to(self, seat_or_point, label="gaze"):
        self._gaze_seq += 1
        seq = self._gaze_seq

        def go():
            if seq != self._gaze_seq:      # a newer gaze target superseded this one
                return
            point = self.seats[seat_or_point] if isinstance(seat_or_point, str) else seat_or_point
            self.robot.attend_location(point)
            self.log.write("gaze", seat_or_point if isinstance(seat_or_point, str) else "group")
        self._enqueue(go, label=label)

    def _idle_gaze(self):
        """Where the robot looks between turns (part of each condition's behaviour)."""
        mode = self.profile.get("idle_gaze", "group")
        if mode == "cycle":
            ids = list(self.seats)
            seat = ids[self._idle_idx % len(ids)]
            self._idle_idx += 1
            self._gaze_to(seat)
        elif mode == "least_active":
            with self.lock:
                seat = self.tracker.least_active()
            self._gaze_to(seat)
        else:
            self._gaze_to(self.centre)

    # ------------------------------------------------------------- invitation
    def invite(self, seat: str | None = None, source: str = "wizard") -> str | None:
        with self.lock:
            if self.condition is None or self._inviting is not None:
                return None
            if seat is None:
                seat = self.tracker.choose(self.profile["policy"])
            self._stop_current()
            self.tracker.register_invite(seat)
            self._inviting = seat
            self._last_invite_t = time.time()
            self._floor_taken.clear()
            self.log.write("invite", seat, source, policy=self.profile["policy"])
        self._enqueue(self._run_invitation, seat, dict(self.profile), label=f"inviting {seat}")
        return seat

    def _run_invitation(self, seat, profile):
        def on_prompt_end():
            with self.lock:
                self.tracker.open_response_window(seat)
        try:
            perform_invitation(self.robot, profile, seat, self.seats, self.centre,
                               responded=self._floor_taken, on_prompt_end=on_prompt_end,
                               emit=self._emit, rng=self.rng)
        finally:
            with self.lock:
                self._inviting = None
                speaker = self.tracker.current
                answered = speaker == seat
        if speaker:
            self.robot.attend_location(self.seats[speaker])
            ack = resolve_gesture(profile.get("acknowledge_response"))
            if answered and ack:
                self.robot.gesture(ack)
                self.log.write("cue", speaker, "acknowledge")

    # ----------------------------------------------------------------- misc
    def look(self, seat: str):
        self._gaze_to(seat.upper() if seat.upper() in self.seats else self.centre)

    def set_auto(self, on: bool):
        self.auto = bool(on)
        self.log.write("note", value=f"auto_invite={'on' if on else 'off'}")

    def note(self, text: str):
        self.log.write("note", value=text)

    def stop_robot(self):
        """Emergency stop: drop queued actions, stop speech, look at group."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._floor_taken.set()
        self.robot.stop_speaking()
        self.robot.attend_location(self.centre)
        self.log.write("note", value="operator_stop")

    def calibrate_from_faces(self) -> dict | None:
        return seats_from_users(self.robot.users(), self.seats)

    def state(self) -> dict:
        with self.lock:
            m = self.tracker.metrics()
            return {
                "group": self.group,
                "condition": self.condition,
                "profile": self.profile.get("label") if self.condition else None,
                "task": self.task,
                "task_title": self.cfg.tasks["tasks"][self.task]["title"] if self.task else None,
                "question_idx": self.question_idx,
                "questions": self.cfg.tasks["tasks"][self.task]["questions"] if self.task else [],
                "current": self.tracker.current,
                "inviting": self._inviting,
                "pending": self.tracker.pending.seat if self.tracker.pending else None,
                "silence_s": round(self.tracker.silence_s(), 1),
                "elapsed_s": round(time.time() - self.condition_started, 1) if self.condition and self.condition_started else 0,
                "auto": self.auto,
                "status": self.status,
                "metrics": m,
                "meta": self.meta,
                "results": self.results,
                "seats": list(self.seats),
                "log_file": str(self.log.path),
                "log_tail": self.log.tail[-12:],
            }

    def shutdown(self):
        self.end_condition()
        self.log.write("session_end", value={"results": self.results})
        # let queued robot actions finish (max 10 s)
        t0 = time.time()
        while not self._q.empty() and time.time() - t0 < 10:
            time.sleep(0.1)
        self._stop.set()
        self.log.close()
