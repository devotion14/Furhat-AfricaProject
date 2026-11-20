"""Discussion controller: WHEN the robot may speak (rules) and WHAT it says (model).

The split is deliberate and is the main design rule of the live discussion mode:

    WHEN  plan_floor()      hand-written rules over perceived state only
                            (who is speaking, silence length, speaking time per
                            person, time since the robot last spoke). The model is
                            never asked whether to speak, so it can never interrupt.
    WHAT  Brain.compose()   MiniMax (Groq as backup) reads the topic and the recent
                            transcript and writes a short, balanced contribution in
                            the language(s) the group is using. Its output is
                            JSON, validated before anything reaches the robot.
                            Invitations are NOT written by the model: they use the
                            fixed C3 cue sequence and prompts from the paper.

Fallbacks: no keys / timeout / unusable reply -> a scripted line (next task
question for a lull, a neutral bridge otherwise). The discussion never stalls.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field

from .llm import extract_json

log = logging.getLogger("nuru.discussion")


# =============================================================================
# Transcript
# =============================================================================
class Transcript:
    def __init__(self):
        self.items: list = []            # Utterance (listener) or RobotTurn

    def add(self, item):
        self.items.append(item)

    def human(self):
        return [u for u in self.items if not getattr(u, "robot", False)]

    def heard_s(self) -> dict:
        """Speaking time per camera person (unlinked voices are not counted here)."""
        out = {}
        for u in self.human():
            if u.person:
                out[u.person] = out.get(u.person, 0.0) + u.duration
        return out

    def human_turns_since_robot(self) -> int:
        n = 0
        for u in reversed(self.items):
            if getattr(u, "robot", False):
                break
            n += 1
        return n

    def last_lang(self, default="en") -> str:
        for u in reversed(self.human()):
            if u.text and u.lang in ("en", "sw"):
                return u.lang
        return default

    def render(self, n: int = 14, names: dict | None = None) -> str:
        """Recent turns as text for the model. People are labelled by a neutral
        name (Person 1, 2, ...) or by their voice when not yet linked to a face."""
        names = names or {}
        lines = []
        for u in self.items[-n:]:
            if getattr(u, "robot", False):
                lines.append(f"Nuru: {u.text}")
            else:
                who = names.get(u.person) or (f"Voice {u.voice}" if u.voice else "Someone")
                said = u.text if u.text else f"(spoke for {u.duration:.0f} s, words not recognised)"
                lines.append(f"{who} [{u.lang or '?'}]: {said}")
        return "\n".join(lines) or "(nobody has spoken yet)"


@dataclass
class RobotTurn:
    text: str
    kind: str
    robot: bool = True


# =============================================================================
# WHEN: floor rules
# =============================================================================
@dataclass
class FloorState:
    someone_speaking: bool
    robot_speaking: bool
    silence_s: float              # since the last human speech ended
    since_robot_s: float          # since the robot last finished speaking
    since_invite_s: float
    elapsed_s: float              # since the discussion started
    human_turns_since_robot: int
    present: list = field(default_factory=list)       # person ids
    dwell_s: dict = field(default_factory=dict)       # person id -> seconds present
    heard_s: dict = field(default_factory=dict)       # person id -> seconds spoken
    last_invited: str | None = None


@dataclass
class FloorDecision:
    action: str                   # wait | respond | lull | invite | close
    target: str | None = None
    reason: str = ""


DEFAULT_RULES = {
    "turn_gap_s": 1.5,          # silence that ends a turn; the robot never speaks before this
    "lull_s": 6.0,              # silence after which the robot brings in a new angle
    "respond_every": 3,         # human turns between robot contributions
    "min_robot_gap_s": 12.0,    # robot does not contribute more often than this
    "invite_gap_s": 35.0,       # between invitations
    "quiet_share": 0.5,         # quiet = below this share of an equal split of speaking time
    "quiet_after_s": 25.0,      # only judge someone quiet after they've been present this long
    "duration_s": 300.0,        # discussion length
}


def quiet_person(s: FloorState, R: dict) -> str | None:
    """Least-heard person whose speaking share is clearly below an equal split."""
    if len(s.present) < 2:
        return None
    total = sum(s.heard_s.get(p, 0.0) for p in s.present)
    fair = total / len(s.present)
    cands = [p for p in s.present
             if s.dwell_s.get(p, 0) >= R["quiet_after_s"]
             and s.heard_s.get(p, 0.0) < R["quiet_share"] * max(fair, 1.0)
             and p != s.last_invited]
    if not cands:
        return None
    return min(cands, key=lambda p: (s.heard_s.get(p, 0.0), -s.dwell_s.get(p, 0)))


def plan_floor(s: FloorState, rules: dict | None = None) -> FloorDecision:
    R = {**DEFAULT_RULES, **(rules or {})}
    if s.someone_speaking:
        return FloorDecision("wait", reason="someone is speaking - never interrupt")
    if s.robot_speaking:
        return FloorDecision("wait", reason="robot is speaking")
    if s.silence_s < R["turn_gap_s"]:
        return FloorDecision("wait", reason=f"turn may not be over ({s.silence_s:.1f} s < {R['turn_gap_s']} s)")
    if s.elapsed_s >= R["duration_s"]:
        return FloorDecision("close", reason="discussion time is up")
    q = quiet_person(s, R)
    if q and s.since_invite_s >= R["invite_gap_s"] and s.since_robot_s >= 4.0:
        return FloorDecision("invite", q, reason=f"{q} heard {s.heard_s.get(q, 0):.0f} s of "
                                                  f"{sum(s.heard_s.values()):.0f} s total")
    if s.silence_s >= R["lull_s"] and s.since_robot_s >= R["lull_s"]:
        return FloorDecision("lull", reason=f"silence {s.silence_s:.0f} s - bring in a new angle")
    if s.human_turns_since_robot >= R["respond_every"] and s.since_robot_s >= R["min_robot_gap_s"]:
        return FloorDecision("respond", reason=f"{s.human_turns_since_robot} turns since the robot spoke")
    return FloorDecision("wait", reason="let the group talk")


# =============================================================================
# WHAT: the model
# =============================================================================
SYSTEM_PROMPT = """You are Nuru, a friendly social robot that facilitates a small-group discussion \
between secondary-school or university students in East Africa. You are BOTH the moderator and a \
participant with views of your own.

How you contribute:
- Balanced opinions: when you give a view, also name a reasonable opposite view, and never present \
one answer as the only right one. Build on what people said; credit ideas ("that point about cost...").
- Keep it SHORT: it is spoken aloud. At most 2 sentences, at most 40 words in total.
- End with an open question to the whole group about half of the time, not always.
- Language: answer in the language the group is using right now. If people mix English and Kiswahili, \
mix the same way (natural code-switching). Use simple Kiswahili and simple English.
- If the discussion has drifted from the topic, gently connect it back.
- The transcript comes from automatic speech recognition and can contain errors; never quote \
garbled words, and if something is unclear, do not guess wildly.
- Never comment on anyone's appearance, never name or single out a person, never pressure anyone \
to speak, never mention that you are an AI model, never give personal data.

Reply with JSON only:
{"kind": "opinion" | "followup" | "summary" | "bridge" | "question",
 "segments": [{"lang": "en" | "sw", "text": "..."}]}
Use one segment per language run, in speaking order (e.g. a Kiswahili phrase then English)."""

ACTION_BRIEF = {
    "respond": "The group just finished a few turns. React to what was said: add a balanced view, "
               "connect two ideas, or ask a follow-up.",
    "lull": "The group has gone quiet. Re-open the discussion: offer a fresh angle or a concrete "
            "example on the topic, then ask an easy open question.",
    "close": "Time is up. Summarise the main ideas you heard in one or two sentences and thank the group.",
    "opening": "Open the discussion: introduce the question below in one or two friendly sentences.",
}


@dataclass
class Contribution:
    kind: str
    segments: list           # [(lang, text)]
    source: str              # model name or 'fallback'

    @property
    def text(self) -> str:
        return " ".join(t for _, t in self.segments)


def validate(obj: dict | None, max_chars: int = 320) -> Contribution | None:
    """Turn model JSON into something the robot may say, or None if unusable."""
    if not obj:
        return None
    segs = obj.get("segments")
    if isinstance(segs, str):
        segs = [{"lang": "en", "text": segs}]
    if not isinstance(segs, list):
        return None
    out = []
    for s in segs[:4]:
        if not isinstance(s, dict):
            continue
        lang = str(s.get("lang", "en")).lower()[:2]
        text = " ".join(str(s.get("text", "")).split()).strip().strip('"')
        if lang not in ("en", "sw") or not text:
            continue
        out.append((lang, text))
    if not out or sum(len(t) for _, t in out) > max_chars:
        return None
    kind = str(obj.get("kind", "opinion"))
    return Contribution(kind if kind in ("opinion", "followup", "summary", "bridge", "question") else "opinion",
                        out, "model")


class Brain:
    def __init__(self, chat, topic: dict, fallbacks: dict, rng: random.Random | None = None,
                 deadline_s: float = 8.0):
        self.chat = chat                      # llm.FallbackChat (or None for offline)
        self.topic = topic                    # task from tasks.json: title, intro, questions
        self.fb = fallbacks
        self.rng = rng or random.Random()
        self.deadline_s = deadline_s
        self._q_i = 0
        self.last_error = None

    def next_question(self) -> str | None:
        qs = self.topic.get("questions", [])
        if self._q_i < len(qs):
            self._q_i += 1
            return qs[self._q_i - 1]
        return None

    def messages(self, action: str, transcript: Transcript, n_people: int, names: dict) -> list:
        qs = "\n".join(f"- {q}" for q in self.topic.get("questions", []))
        user = (f"Topic: {self.topic.get('title', '')}\n{self.topic.get('intro', '')}\n"
                f"Discussion questions:\n{qs}\n\n"
                f"People present: {n_people}\n\nRecent discussion:\n{transcript.render(names=names)}\n\n"
                f"Group's current language: {transcript.last_lang()}\n\n"
                f"Your task now: {ACTION_BRIEF[action]}")
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def compose(self, action: str, transcript: Transcript, n_people: int, names: dict | None = None) -> Contribution:
        """Always returns something sayable (model output or a scripted fallback)."""
        if self.chat is not None and self.chat.available:
            reply = self.chat.chat(self.messages(action, transcript, n_people, names or {}),
                                   deadline_s=self.deadline_s)
            c = validate(extract_json(reply))
            if c:
                c.source = getattr(self.chat, "used", None) or "model"
                return c
            self.last_error = "no reply / timeout" if reply is None else f"unusable reply: {reply[:120]!r}"
            log.warning("model contribution failed: %s", self.last_error)
        return self.fallback(action, transcript)

    def fallback(self, action: str, transcript: Transcript) -> Contribution:
        lang = transcript.last_lang()
        if action in ("lull", "opening"):
            q = self.next_question()
            if q:
                return Contribution("question", [("en", q)], "fallback")
        pool = self.fb.get(action) or self.fb.get("respond") or [{"en": "Interesting. Does anyone see it differently?"}]
        line = self.rng.choice(pool)
        segs = [(lang, line[lang])] if lang in line else [(k, v) for k, v in line.items() if k in ("sw", "en")]
        return Contribution("bridge", segs, "fallback")


def contribution_log(c: Contribution) -> str:
    return json.dumps({"kind": c.kind, "segments": c.segments, "source": c.source}, ensure_ascii=False)
