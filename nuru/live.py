"""Autonomous live mode:  python -m nuru live   (add --mock --scenario group to rehearse)

The robot runs the whole interaction on its own. Nobody presses a button.

    SENSE  (nuru/perception.py)      camera faces -> Percept: who is present, distance,
                                     how long each has been there, who arrived/left
           (robot.listen_timed)      microphone -> spoke / silent / error + duration
    ------------------------------------------------------------------------------------
    PLAN   (plan_* functions below)  pure functions: Percept + Memory -> Decision.
                                     No robot calls in here, so every rule is unit-tested.
    ------------------------------------------------------------------------------------
    ACT    (LiveSession.act_* below, nuru/cues.py)
                                     Decision -> speech (en/sw), gaze, head, gestures, LED

The social feature is perceived, not said: HOW MANY people are present (and
who has been there longest without being heard). One person gets direct
one-to-one cues; two or more get the culturally adapted Nuru group cues
(C3 profile): group first, then the pair, short individual gaze with a soft
nod, bilingual prompt, gaze back to the group, and a gentle release that never
insists. A newcomer is welcomed; when people leave the robot adapts or ends.

Failure handling by design (see plan_after_listen):
  * silence      -> solo: "Sorry, I didn't catch that", one retry, then move on
                    group: gentle release "Hakuna shida", invite nobody twice
  * ASR / network error -> "Sorry, I could not hear properly", move on
  * every robot call has a timeout and never raises; listening has a hard cap
  * camera fails -> keep last known people; everyone leaves -> end politely
  * the LED shows the robot's state (blue = listening), so the person always
    has feedback while the robot is waiting
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from dataclasses import dataclass, field

from .cues import CUSTOM_GESTURES, lerp, nearest_neighbour, pick_prompt, pick_release, resolve_gesture
from .eventlog import EventLog
from .perception import PeopleTracker, Percept

log = logging.getLogger("nuru.live")


# =============================================================================
# PLAN: pure decision rules. Input = Percept (+ Memory, + listen result).
# =============================================================================
@dataclass
class Decision:
    action: str               # idle | watch | beckon | start | welcome | farewell | ask | invite |
                              # acknowledge | repair | repair_tech | repair_tech_move_on | release
    target: str | None = None # person id
    mode: str | None = None   # solo | group
    reason: str = ""


@dataclass
class Memory:
    """What the robot remembers within one interaction (reset when everyone leaves)."""
    greeted: set = field(default_factory=set)
    heard_s: dict = field(default_factory=dict)          # person id -> seconds heard after being addressed
    invited: list = field(default_factory=list)          # person ids, in order
    group_heard_s: float = 0.0                           # speech after open questions (speaker unknown)
    beckoned: set = field(default_factory=set)


def plan_mode(p: Percept) -> str:
    """THE social adaptation: group size, perceived by the camera."""
    return "solo" if p.n == 1 else "group"


def plan_engagement(p: Percept, mem: Memory, engage_m: float) -> Decision:
    """Before the interaction: should the robot start, and how?"""
    if not p.camera_ok and p.n == 0:
        return Decision("idle", reason="camera not answering")
    if p.n == 0:
        return Decision("idle", reason="nobody present")
    near = p.nearest()
    if near.distance > engage_m:
        if near.id in mem.beckoned:
            return Decision("watch", near.id, reason=f"nearest person at {near.distance:.1f} m, already beckoned")
        return Decision("beckon", near.id, reason=f"nearest person at {near.distance:.1f} m > {engage_m} m")
    mode = plan_mode(p)
    return Decision("start", near.id, mode, reason=f"{p.n} person(s) within {engage_m} m -> {mode} mode")


def plan_between_questions(p: Percept, mem: Memory) -> Decision:
    """Checked before every question: did anyone arrive or leave?"""
    if p.n == 0:
        return Decision("farewell", reason="everyone left")
    new = [x for x in p.people if x.id not in mem.greeted]
    if new:
        return Decision("welcome", new[0].id, plan_mode(p), reason=f"{new[0].id} arrived {p.dwell_s(new[0].id):.0f} s ago")
    return Decision("ask", mode=plan_mode(p), reason=f"{p.n} present -> {plan_mode(p)} mode")


def plan_invitee(p: Percept, mem: Memory) -> str | None:
    """Group mode: whom to invite. Least heard first; among equals, the one who
    has been present longest (quiet for longest). Never the same person twice
    in a row, so nobody is singled out."""
    if not p.people:
        return None
    last = mem.invited[-1] if mem.invited else None
    candidates = [x for x in p.people if x.id != last] or p.people
    return min(candidates, key=lambda x: (round(mem.heard_s.get(x.id, 0.0), 1), -p.dwell_s(x.id))).id


def plan_after_listen(status: str, mode: str, attempt: int, invited: bool) -> Decision:
    """What to do with the listen result. Uses only spoke/silent/error."""
    if status == "speech":
        return Decision("acknowledge", mode=mode, reason="person spoke")
    if status == "error":
        if mode == "solo" and attempt == 0:
            return Decision("repair_tech", mode=mode, reason="speech recognition / network failed -> retry once")
        return Decision("repair_tech_move_on", mode=mode, reason="speech recognition / network failed -> move on")
    if mode == "group":
        if invited:   # C3: never insist on an individual
            return Decision("release", mode=mode, reason="invitee silent -> gentle release")
        return Decision("invite", mode=mode, reason="open floor silent -> invite someone")
    if attempt == 0:
        return Decision("repair", mode=mode, reason="silence -> repair once")
    return Decision("release", mode=mode, reason="silent again -> move on")


# =============================================================================
# ACT: turning decisions into robot behaviour, and running the loop.
# =============================================================================
class LiveSession:
    def __init__(self, cfg, robot, sense: PeopleTracker | None = None, seed: int | None = None,
                 verbose: bool = True, speed: float = 1.0):
        self.cfg, self.robot = cfg, robot
        self.L = cfg.live
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.speed = speed
        self.sense = sense or PeopleTracker(robot, confirm_s=self.L.get("confirm_s", 1.0),
                                            lost_after_s=self.L.get("lost_after_s", 3.0),
                                            max_distance_m=self.L.get("max_distance_m", 2.5))
        self.log = EventLog(cfg.log_dir, "LIVE")
        self.stop_event = threading.Event()
        self.mem = Memory()
        self.last: Percept = Percept(time.time())
        self._scan_i = 0
        self.n_sessions = 0

    # -- helpers -------------------------------------------------------------
    def _print(self, tag, msg):
        if self.verbose:
            print(f"  [{tag:5}] {msg}", flush=True)

    def _sleep(self, s):
        self.stop_event.wait(s * self.speed)

    def percept(self) -> Percept:
        p = self.sense.snapshot()
        ids = [x.id for x in p.people]
        if ids != [x.id for x in self.last.people] or p.arrived or p.left:
            desc = ", ".join(f"{x.id} {x.distance:.1f} m/{p.dwell_s(x.id):.0f} s" for x in p.people) or "nobody"
            self._print("SENSE", f"{p.n} present: {desc}" + ("" if p.camera_ok else "  (camera not answering)"))
            self.log.write("percept", value={"n": p.n, "people": ids, "arrived": p.arrived, "left": p.left,
                                             "camera_ok": p.camera_ok})
        self.last = p
        return p

    def _decided(self, d: Decision):
        self._print("PLAN", f"{d.action}{' -> ' + d.target if d.target else ''}  ({d.reason})")
        self.log.write("decision", d.target or "", d.action, mode=d.mode, reason=d.reason)

    def led(self, state):
        rgb = (self.L.get("led") or {}).get(state)
        if rgb:
            self.robot.led(*rgb)

    def say_line(self, key, langs=("sw", "en")):
        line = (self.L.get("lines") or {}).get(key, {})
        self.led("speaking")
        for lang in langs:
            if line.get(lang):
                self._print("ACT", f"say ({lang}) {line[lang]}")
                self.log.write("robot_say", value=line[lang], lang=lang)
                self.robot.say(line[lang], lang=lang, blocking=True)

    def look(self, point, what=""):
        self.robot.attend_location(point)
        if what:
            self._print("ACT", f"look at {what}")

    def centre(self, p: Percept):
        if not p.people:
            return self.cfg.group_centre
        n = len(p.people)
        return tuple(sum(x.location[i] for x in p.people) / n for i in range(3))

    def listen(self, mode, who="") -> dict:
        self.led("listening")
        self._print("SENSE", f"listening{' to ' + who if who else ''} (max {self.L.get('listen_timeout_s', 20)} s)")
        r = self.robot.listen_timed(self.L.get("listen_language", "en-US"), float(self.L.get("listen_timeout_s", 20)))
        self.led("idle")
        self._print("SENSE", f"heard: {r['status']} ({r['duration_s']:.1f} s)")
        self.log.write("listen", who, r["status"], duration_s=round(r["duration_s"], 2), mode=mode)
        return r

    # -- act: one action per decision ----------------------------------------
    def act_idle(self):
        """Nobody there: slow look-around so the robot appears alive and finds people."""
        self.led("idle")
        xs = (-0.6, 0.0, 0.6, 0.0)
        self.robot.attend_location((xs[self._scan_i % len(xs)], 0.0, 1.5))
        self._scan_i += 1

    def act_beckon(self, p: Percept, pid: str):
        self.mem.beckoned.add(pid)
        x = p.get(pid)
        self.led("attention")
        self.look(x.location, f"{pid} (far away)")
        self.robot.gesture(CUSTOM_GESTURES["NuruWarmSmile"])
        self.say_line("beckon")

    def act_greet(self, p: Percept, mode: str):
        self.led("attention")
        if mode == "solo":
            x = p.people[0]
            self.look(x.location, f"{x.id} (direct, one-to-one)")
            self.robot.gesture(CUSTOM_GESTURES["NuruWarmSmile"])
            self.say_line("greet_solo")
        else:
            # Sweep: a short glance at each person, left to right, then the group.
            for x in p.people:
                self.look(x.location, f"{x.id} (greeting sweep)")
                self._sleep(0.5)
            self.look(self.centre(p), "the group")
            self.robot.gesture(CUSTOM_GESTURES["NuruSoftNod"])
            self.say_line("greet_group")
        self.mem.greeted.update(x.id for x in p.people)

    def act_welcome(self, p: Percept, pid: str):
        x = p.get(pid)
        self.look(x.location, f"newcomer {pid}")
        self.robot.gesture(CUSTOM_GESTURES["NuruSoftNod"], blocking=True)
        self.say_line("welcome")
        self.mem.greeted.add(pid)
        if p.n > 1:
            self.look(self.centre(p), "the group")

    def act_acknowledge(self, p: Percept, pid: str | None, heard_s: float, mode: str):
        """Immediate feedback that the robot heard the person (content is not used)."""
        if pid:
            self.mem.heard_s[pid] = self.mem.heard_s.get(pid, 0.0) + heard_s
            x = p.get(pid)
            if x:
                self.look(x.location)
        else:
            self.mem.group_heard_s += heard_s
        nod = "NuruSoftNod" if mode == "group" else "Nod"
        self.robot.gesture(resolve_gesture(nod), blocking=False)
        self.say_line("ack_group" if (mode == "group" and not pid) else "ack")

    def act_invite(self, p: Percept, pid: str, profile: dict):
        """C3 invitation with LIVE face positions instead of fixed seats.
        Same cue order as cues.perform_invitation; the listening part is ours."""
        seats = {x.id: x.location for x in p.people}
        centre = self.centre(p)
        self.mem.invited.append(pid)
        self.led("attention")
        self.log.write("invite", pid, "live", policy="least_heard")
        if profile.get("group_glance_s", 0) > 0:
            self.look(centre, "the group (group glance)")
            self._sleep(profile["group_glance_s"])
        nb = nearest_neighbour(pid, seats)
        if profile.get("pair_glance_s", 0) > 0 and nb:
            self.look(lerp(seats[pid], seats[nb], 0.5), f"pair {pid}+{nb}")
            self._sleep(profile["pair_glance_s"])
        self.look(seats[pid], f"{pid} (short individual gaze)")
        if profile.get("smile"):
            self.robot.gesture(CUSTOM_GESTURES["NuruWarmSmile"])
        nod = resolve_gesture(profile.get("nod"))
        if nod:
            self.robot.gesture(nod, blocking=True)
        self.led("speaking")
        for lang, text in pick_prompt(profile, self.rng):
            self._print("ACT", f"say ({lang}) {text}")
            self.log.write("robot_say", pid, text, lang=lang)
            self.robot.say(text, lang=lang, blocking=True)
        self._sleep(float(profile.get("gaze_hold_s", 1.5)))
        if profile.get("gaze_during_wait") == "group":
            self.look(centre, "the group (takes pressure off)")

    def act_release(self, p: Percept, mode: str, profile: dict):
        if mode == "group":
            self.look(self.centre(p), "the group")
            for lang, text in pick_release(profile, self.rng):
                self._print("ACT", f"say ({lang}) {text}")
                self.robot.say(text, lang=lang, blocking=True)
        else:
            self.say_line("release_solo")

    # -- the interaction -----------------------------------------------------
    def ask(self, p: Percept, question: str, mode: str):
        group_prof = self.cfg.profile(self.L.get("group_profile", "c3"))
        if mode == "solo":
            x = p.people[0]
            solo = self.L.get("solo", {})
            self.look(x.location, f"{x.id} (direct gaze)")
            if solo.get("smile"):
                self.robot.gesture(CUSTOM_GESTURES["NuruWarmSmile"])
            self.led("speaking")
            self._print("ACT", f"say (en) {question}")
            self.log.write("task_question", x.id, question, mode=mode)
            self.robot.say(question, "en", blocking=True)
            for attempt in range(2):
                r = self.listen(mode, x.id)
                if self.percept().n == 0 and r["status"] != "speech":
                    self._decided(Decision("farewell", reason="person left while the robot was listening"))
                    return
                d = plan_after_listen(r["status"], mode, attempt, invited=True)
                self._decided(d)
                if d.action == "acknowledge":
                    self.act_acknowledge(self.last, x.id, r["duration_s"], mode)
                    return
                if d.action in ("repair", "repair_tech"):
                    self.say_line(d.action, ("en",))
                    self.say_line_text(question)          # repeat the question once
                    continue
                if d.action == "repair_tech_move_on":
                    self.say_line("repair_tech", ("en",))
                    self.say_line("release_solo", ("en",))
                    return
                self.act_release(self.last, mode, group_prof)
                return
            return

        # group: open question to everyone, then invite the least-heard person
        self.look(self.centre(p), "the group")
        self.led("speaking")
        self._print("ACT", f"say (en) {question}")
        self.log.write("task_question", "", question, mode=mode)
        self.robot.say(question, "en", blocking=True)
        r = self.listen(mode, "the group")
        d = plan_after_listen(r["status"], mode, 0, invited=False)
        self._decided(d)
        if d.action == "acknowledge":
            self.act_acknowledge(self.last, None, r["duration_s"], mode)
        elif d.action == "repair_tech_move_on":
            self.say_line("repair_tech", ("en",))
        p = self.percept()
        if p.n < 2:                                  # someone left meanwhile
            return
        pid = plan_invitee(p, self.mem)
        heard = ", ".join(f"{x.id} heard {self.mem.heard_s.get(x.id, 0):.1f} s/present {p.dwell_s(x.id):.0f} s" for x in p.people)
        self._decided(Decision("invite", pid, mode, reason=f"least heard, then longest present: {heard}"))
        self.act_invite(p, pid, group_prof)
        r = self.listen(mode, pid)
        d = plan_after_listen(r["status"], mode, 0, invited=True)
        self._decided(d)
        if d.action == "acknowledge":
            self.act_acknowledge(self.last, pid, r["duration_s"], mode)
        elif d.action == "repair_tech_move_on":
            self.say_line("repair_tech", ("en",))
            self.act_release(self.last, mode, group_prof)
        else:
            self.act_release(self.last, mode, group_prof)

    def interaction(self, p: Percept, mode: str):
        self.n_sessions += 1
        self.log.write("interaction_start", value={"mode": mode, "n": p.n})
        self._print("PLAN", f"=== interaction {self.n_sessions} starts in {mode.upper()} mode ===")
        self.act_greet(p, mode)
        if self.L.get("interaction") == "discussion" and getattr(self, "listener", None) is not None:
            self.discuss(p, mode)
            self.log.write("interaction_end", value={"heard_s": self.mem.heard_s, "invited": self.mem.invited})
            return
        task = self.cfg.tasks["tasks"][self.L.get("task", "T3")]
        self.look(self.centre(p), "")
        self.say_line_text(task["intro"])
        for q in task["questions"][: int(self.L.get("n_questions", 2))]:
            if self.stop_event.is_set():
                return
            p = self.percept()
            d = plan_between_questions(p, self.mem)
            if d.action == "farewell":
                self._decided(d)
                return
            while d.action == "welcome":
                self._decided(d)
                self.act_welcome(p, d.target)
                d = plan_between_questions(p, self.mem)
            if d.mode != mode:
                self._print("PLAN", f"group size changed -> switching {mode} -> {d.mode} mode")
                self.log.write("decision", "", "switch_mode", frm=mode, to=d.mode)
                mode = d.mode
            self._decided(d)
            self.ask(p, q, mode)
        p = self.percept()
        if p.n:
            self.look(self.centre(p), "everyone")
            self.robot.gesture(CUSTOM_GESTURES["NuruSoftNod"])
            self.say_line("closing_solo" if p.n == 1 else "closing_group")
        self.log.write("interaction_end", value={"heard_s": self.mem.heard_s, "invited": self.mem.invited})

    def say_line_text(self, text, lang="en"):
        self.led("speaking")
        self._print("ACT", f"say ({lang}) {text}")
        self.log.write("robot_say", value=text, lang=lang)
        self.robot.say(text, lang=lang, blocking=True)


    # =========================================================================
    # DISCUSSION MODE (live.json "interaction": "discussion")
    #   SENSE  faces (perception.py) + passive listening (listener.py):
    #          what was said, in which language, by which voice -> which person
    #   PLAN   discussion.plan_floor: WHEN to speak (rules, never interrupts)
    #          discussion.Brain:      WHAT to say (MiniMax -> Groq -> scripted)
    #   ACT    speech in en/sw segments, gaze, nods, LED; C3 cues for invitations
    # =========================================================================
    def setup_discussion(self, listener=None):
        from .discussion import Transcript
        from .llm import Transcriber, make_chat
        from .voices import Embedder, SpeakerLinker, VoiceBank
        D = self.D = self.L.get("discussion", {})
        self.transcript = Transcript()
        self.linker = SpeakerLinker()
        self.bank = None
        self._utt_q: queue.Queue = queue.Queue()
        self._pending_invite = None                  # (person id, time)
        self.chat = make_chat(self.L)
        if listener is None:
            from .listener import Listener
            emb = Embedder(prefer_neural=D.get("neural_voiceprints", True))
            self.bank = VoiceBank(emb.threshold)
            stt = Transcriber(self.L)
            listener = Listener(self._utt_q.put, stt, emb, self.bank,
                                threshold_db=float(D.get("vad_threshold_db", 10.0)))
            if D.get("audio_source", "furhat") == "furhat":
                listener.start_furhat(self.cfg.host, int(D.get("mic_channel", 0)), int(D.get("robot_channel", 1)))
            else:
                listener.start_mic(D.get("mic_device"))
            self._print("INFO", f"listening: {D.get('audio_source', 'furhat')} audio, voiceprints {emb.kind}, "
                                f"speech-to-text {'Groq Whisper' if stt.available else 'OFF (no GROQ key)'}")
        self.listener = listener
        self.cam = None
        if D.get("camera_speaker", True) and not getattr(listener, "scripted", False):
            try:
                from .speaker_cam import CameraSpeakerHint
                self.cam = CameraSpeakerHint(self.cfg, D)
                self._print("INFO", "camera speaker hint: on (mouth movement links voices to faces)")
            except Exception as e:                  # mediapipe/opencv/zmq missing: voiceprints only
                self._print("INFO", f"camera speaker hint: off ({e})")
        models = [c.name for c in self.chat.clients]
        self._print("INFO", f"discussion model: {' -> '.join(models) if models else 'NONE (scripted lines only)'}")
        self.robot = _GatedRobot(self.robot, listener)

    def _scale(self, secs: float) -> float:
        """Real seconds -> scenario seconds (mock runs faster with speed < 1)."""
        return secs / self.speed if self.speed else secs

    def _drain_utterances(self, p: Percept):
        present = [x.id for x in p.people]
        while True:
            try:
                u = self._utt_q.get_nowait()
            except queue.Empty:
                return
            invited = None
            if self._pending_invite and self._scale(u.end - self._pending_invite[1]) <= \
                    float(self.D.get("response_window_s", 8.0)) + 15:
                invited = self._pending_invite[0]
            if u.person is None:
                seen = None
                if self.cam is not None and self.cam.available:
                    seen = self.cam.person_for(self.cam.seat_during(u.start, u.end), p.people)
                u.person, u.how = self.linker.link(u.voice, present, invited, seen=seen)
            self.transcript.add(u)
            who = u.person or f"voice {u.voice or '?'}"
            said = u.text if u.text else "(words not recognised)"
            self._print("SENSE", f"heard {who} ({u.voice or '-'}, {u.how}) [{u.lang or '?'}] "
                                 f"{self._scale(u.duration):.1f} s: {said}")
            self.log.write("utterance", u.person or "", u.text if self.D.get("log_text", True) else "",
                           voice=u.voice, lang=u.lang, duration_s=round(self._scale(u.duration), 2), how=u.how)
            if u.person:
                self.mem.heard_s[u.person] = self.mem.heard_s.get(u.person, 0.0) + self._scale(u.duration)
            if invited and u.person == invited:
                self._pending_invite = None
                x = p.get(invited)
                if x:
                    self.look(x.location)
                self.robot.gesture(resolve_gesture("NuruSoftNod"), blocking=False)   # backchannel: "I hear you"
                self.look(self.centre(p))

    def _floor_state(self, p: Percept, t0, last_robot, last_invite):
        from .discussion import FloorState
        now = time.time()
        return FloorState(
            someone_speaking=self.listener.someone_speaking,
            robot_speaking=self.listener.robot_speaking,
            silence_s=self._scale(self.listener.silence_s()),
            since_robot_s=self._scale(now - last_robot),
            since_invite_s=self._scale(now - last_invite),
            elapsed_s=self._scale(now - t0),
            human_turns_since_robot=self.transcript.human_turns_since_robot(),
            present=[x.id for x in p.people],
            dwell_s={x.id: self._scale(p.dwell_s(x.id)) for x in p.people},
            heard_s=dict(self.mem.heard_s),
            last_invited=self.mem.invited[-1] if self.mem.invited else None,
            robot_turns_since_human=self.transcript.robot_turns_since_human(),
            unattributed_s=self._scale(self.transcript.unattributed_s()))

    def _names(self, p: Percept) -> dict:
        return {x.id: f"Person {i + 1}" for i, x in enumerate(p.people)}

    def think(self, action: str, p: Percept, brain, allow_abort: bool = True):
        """Ask the model in the background. While waiting the robot shows it is
        thinking (LED + look away briefly + filler if slow). If someone starts
        talking meanwhile, the contribution is dropped: the robot never interrupts."""
        box = {}
        th = threading.Thread(target=lambda: box.setdefault("c", brain.compose(
            action, self.transcript, p.n, self._names(p))), daemon=True)
        th.start()
        self.led("thinking")
        t0, filled = time.time(), False
        while th.is_alive():
            if allow_abort and self.listener.someone_speaking:
                self._print("PLAN", "someone started talking while the robot was thinking -> drop it, keep listening")
                self.log.write("decision", "", "yield", reason="speech during thinking")
                self.led("idle")
                return None
            if not filled and self._scale(time.time() - t0) > float(self.D.get("filler_after_s", 1.5)):
                filled = True
                c = self.centre(p)
                self.robot.attend_location((c[0] + 0.3, c[1] + 0.25, c[2]))      # "thinking" glance up
                self.robot.gesture(resolve_gesture("NuruSoftNod"), blocking=False)
            th.join(0.1)
        self.led("idle")
        return box.get("c")

    def speak(self, c, p: Percept, to_group: bool = True):
        from .discussion import RobotTurn, contribution_log
        if to_group:
            self.look(self.centre(p), "the group")
        if c.kind in ("opinion", "bridge"):
            self.robot.gesture(resolve_gesture("NuruSoftNod"), blocking=False)
        self.led("speaking")
        self._print("ACT", f"[{c.kind} via {c.source}]")
        for lang, text in c.segments:
            self._print("ACT", f"say ({lang}) {text}")
            self.robot.say(text, lang=lang, blocking=True)
        self.led("idle")
        self.log.write("robot_contribution", "", contribution_log(c))
        self.transcript.add(RobotTurn(c.text, c.kind))

    def discuss(self, p: Percept, mode: str):
        from .discussion import Brain, Contribution, plan_floor
        D = self.D
        rules = dict(D.get("rules", {}))
        if mode == "solo":
            rules.update(D.get("solo_rules", {}))
        task = self.cfg.tasks["tasks"][D.get("task", self.L.get("task", "T3"))]
        brain = Brain(self.chat, task, D.get("fallback_lines", {}), self.rng, float(D.get("model_deadline_s", 8.0)))
        group_prof = self.cfg.profile(self.L.get("group_profile", "c3"))
        while not self._utt_q.empty():                 # forget anything heard before the greeting
            self._utt_q.get_nowait()
        if self.bank is not None:
            self.bank.reset()
        self.linker.reset()
        self.transcript.items.clear()

        # opening: the task intro and first question are scripted (reliable, no model latency)
        self.look(self.centre(p), "the group")
        self.say_line_text(task["intro"])
        self.speak(Contribution("question", [("en", brain.next_question() or task["intro"])], "script"), p)
        t0 = last_robot = time.time()
        last_invite = 0.0
        last_reason = None
        while not self.stop_event.is_set():
            p = self.percept()
            if p.n == 0:
                self._decided(Decision("farewell", reason="everyone left"))
                return
            for x in [x for x in p.people if x.id not in self.mem.greeted]:
                self._decided(Decision("welcome", x.id, reason="arrived during the discussion"))
                self.act_welcome(p, x.id)
                last_robot = time.time()
            self._drain_utterances(p)

            # an invitation nobody answered -> gentle release (C3: never insist)
            if self._pending_invite and not self.listener.someone_speaking and \
                    self._scale(time.time() - self._pending_invite[1]) > float(D.get("response_window_s", 8.0)):
                pid = self._pending_invite[0]
                self._pending_invite = None
                self._decided(Decision("release", pid, mode, reason="invitee did not answer"))
                self.log.write("no_response", pid)
                self.act_release(p, "group", group_prof)
                last_robot = time.time()
                continue

            if self._pending_invite:                   # waiting for the invitee: keep the floor open
                self._sleep(0.25)
                continue
            d = plan_floor(self._floor_state(p, t0, last_robot, last_invite), rules)
            if d.action == "wait":
                if d.reason != last_reason and "speaking" not in d.reason:
                    self.log.write("floor", "", "wait", reason=d.reason)
                last_reason = d.reason
                self._sleep(0.25)
                continue
            last_reason = None
            if d.action == "invite":
                if self._pending_invite:
                    self._sleep(0.25)
                    continue
                self._decided(Decision("invite", d.target, mode, d.reason))
                self.act_invite(p, d.target, group_prof)
                self._pending_invite = (d.target, time.time())
                last_invite = last_robot = time.time()
                continue
            self._decided(Decision(d.action, mode=mode, reason=d.reason))
            c = self.think(d.action, p, brain, allow_abort=d.action != "close")
            if c is None:
                continue
            if c.source == "fallback":
                self.log.write("model_fallback", "", d.action, error=brain.last_error)
            self.speak(c, p)
            last_robot = time.time()
            if d.action == "close":
                self.robot.gesture(CUSTOM_GESTURES["NuruSoftNod"])
                self.say_line("closing_solo" if p.n == 1 else "closing_group")
                return

    def wait_until_empty(self, max_s=60.0):
        """After an interaction, do not re-greet the same people: wait until they leave."""
        t0 = time.time()
        self.led("idle")
        while not self.stop_event.is_set() and time.time() - t0 < max_s * max(self.speed, 0.05):
            p = self.percept()
            if p.n == 0:
                return
            near = p.nearest()
            self.robot.attend_location(near.location)    # keeps following, but does not restart
            self._sleep(1.0)

    def run(self, max_interactions: int | None = None):
        self.sense.start()
        if self.L.get("interaction") == "discussion" and getattr(self, "listener", None) is None:
            try:
                self.setup_discussion()
            except Exception as e:                     # no audio feed / no zmq: keep the Lab 1 flow
                log.exception("discussion setup failed")
                self._print("ERROR", f"discussion mode unavailable ({e}) - using the question flow")
                self.L = {**self.L, "interaction": "questions"}
        self._print("INFO", "live mode: waiting for people (Ctrl+C to stop)")
        self.log.write("session_start", value={"mode": "live"})
        try:
            while not self.stop_event.is_set():
                try:
                    p = self.percept()
                    d = plan_engagement(p, self.mem, float(self.L.get("engage_distance_m", 1.6)))
                    if d.action == "idle":
                        if p.n == 0 and self.mem.greeted:
                            self.mem = Memory()            # forget everyone once they are gone
                        self.act_idle()
                        self._sleep(2.0)
                    elif d.action == "watch":
                        self.robot.attend_location(p.get(d.target).location)
                        self._sleep(0.5)
                    elif d.action == "beckon":
                        self._decided(d)
                        self.act_beckon(p, d.target)
                    else:
                        self._decided(d)
                        self.mem = Memory()
                        self.interaction(p, d.mode)
                        if max_interactions and self.n_sessions >= max_interactions:
                            break
                        self.wait_until_empty()
                except Exception as e:                    # the loop must never die mid-demo
                    log.exception("live loop error")
                    self.log.write("robot_error", value=str(e))
                    self._print("ERROR", f"{e} - recovering")
                    self._sleep(1.0)
        finally:
            self.sense.stop()
            if getattr(self, "listener", None) is not None:
                self.listener.stop()
            if getattr(self, "cam", None) is not None:
                self.cam.stop()
            self.led("idle")
            self.log.write("session_end")
            self.log.close()
            self._print("INFO", f"log: {self.log.path}")


class _GatedRobot:
    """Wraps the robot so the listener ignores the mic while the robot speaks
    (needed with a laptop mic; with the Furhat audio feed the robot channel
    does this as well)."""

    def __init__(self, robot, listener):
        self._robot, self._listener = robot, listener

    def say(self, text, lang="en", blocking=True):
        self._listener.set_robot_speaking(True)
        try:
            return self._robot.say(text, lang=lang, blocking=blocking)
        finally:
            self._listener.set_robot_speaking(False)

    def __getattr__(self, name):
        return getattr(self._robot, name)


# =============================================================================
# Rehearsal scenarios for --mock (the camera and microphone are scripted)
# =============================================================================
def _face(pid, x, z):
    return {"id": pid, "location": {"x": x, "y": 0.0, "z": z}}


SCENARIOS = {
    # (time s, scene)   +  listen outcomes in order
    "solo": ([(1, [_face("p1", 0.0, 1.0)]), (40, [])],
             [("speech", 2.0), ("silence", 0), ("silence", 0)]),
    "group": ([(1, [_face("p1", -0.4, 1.1), _face("p2", 0.4, 1.1)]), (60, [])],
              [("speech", 3.0), ("silence", 0), ("speech", 1.5), ("speech", 2.0)]),
    "newcomer": ([(1, [_face("p1", 0.0, 1.0)]), (14, [_face("p1", 0.0, 1.0), _face("p2", 0.5, 1.1)]), (70, [])],
                 [("speech", 2.0), ("speech", 2.0), ("silence", 0), ("speech", 1.0)]),
    "far": ([(1, [_face("p1", 0.2, 2.2)]), (25, [_face("p1", 0.1, 1.1)]), (70, [])],
            [("speech", 1.5), ("speech", 1.5)]),
    "failures": ([(1, [_face("p1", 0.0, 1.0)]), (40, [])],
                 [("error", 0), ("speech", 1.0), ("silence", 0), ("silence", 0)]),
}


def play_scenario(robot, name: str, speed: float, stop: threading.Event):
    timeline, heard = SCENARIOS[name]
    robot.heard = list(heard)

    def run():
        t0 = time.time()
        for t, scene in timeline:
            while time.time() - t0 < t * speed:
                if stop.wait(0.05):
                    return
            robot.scene = scene
            print(f"  [scene] t={t}s: {len(scene)} face(s) {[f['id'] for f in scene]}", flush=True)
    threading.Thread(target=run, daemon=True, name="nuru-scenario").start()


# Discussion rehearsal (--mock --scenario discussion): p1 talks a lot, p2 only when invited.
DISCUSSION_SCRIPT = [
    ("wait_robot", 1),
    ("say", "V1", "en", "I think we should go to Mombasa, the students can learn about the coast and its history.", 5),
    ("pause", 1),
    ("say", "V1", "sw", "Pia Fort Jesus ni mahali pazuri sana kujifunza historia.", 4),
    ("pause", 1),
    ("say", "V1", "en", "And it is not too expensive if we take the train.", 3),
    ("wait_robot", 2),
    ("say", "V1", "en", "Yes, but some parents might worry about the cost and the long trip.", 4),
    ("wait_invite", "p2"),
    ("pause", 2),
    ("say", "V2", "sw", "Mimi nafikiri tunaweza kwenda Nairobi National Park, ni karibu na ni rahisi.", 5),
    ("pause", 1.8),
    ("say", "V2", "en", "Yeah, and we can learn about wildlife conservation there.", 3),
    ("pause", 10),
    ("wait_robot", 4),
    ("say", "V1", "en", "Maybe we can vote between the two places.", 3),
    ("pause", 15),
]


def play_discussion(sess, robot, listener, speed: float, stop: threading.Event):
    """Scripted camera + voices for the discussion rehearsal."""
    from .discussion import RobotTurn

    def wait(cond, max_s):
        t_end = time.time() + max_s * speed
        while not cond() and time.time() < t_end:
            if stop.wait(0.05):
                return False
        return True

    def run():
        if stop.wait(1 * speed):
            return
        robot.scene = [_face("p1", -0.4, 1.1), _face("p2", 0.4, 1.1)]
        print("  [scene] 2 faces ['p1', 'p2']", flush=True)
        for step in DISCUSSION_SCRIPT:
            if stop.is_set():
                return
            kind = step[0]
            if kind == "wait_robot":
                wait(lambda: sum(isinstance(u, RobotTurn) for u in sess.transcript.items) >= step[1], 60)
            elif kind == "wait_invite":
                wait(lambda: sess._pending_invite and sess._pending_invite[0] == step[1], 60)
            elif kind == "pause":
                stop.wait(step[1] * speed)
            else:
                _, voice, lang, text, secs = step
                print(f"  [human] {voice} ({lang}): {text}", flush=True)
                listener.say(text, lang, voice, secs)
        wait(lambda: False, 200)                    # stay until the robot closes
        robot.scene = []
        print("  [scene] everyone left", flush=True)
    threading.Thread(target=run, daemon=True, name="nuru-discussion-scenario").start()
