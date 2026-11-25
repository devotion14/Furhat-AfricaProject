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

    def robot_turns_since_human(self) -> int:
        n = 0
        for u in reversed(self.items):
            if not getattr(u, "robot", False):
                break
            n += 1
        return n

    def unattributed_s(self) -> float:
        return sum(u.duration for u in self.human() if not u.person)

    def language_mix(self, n: int = 8) -> dict:
        """Languages of the last n human turns, e.g. {'en': 5, 'sw': 2}."""
        out = {}
        for u in self.human()[-n:]:
            if u.text and u.lang:
                out[u.lang] = out.get(u.lang, 0) + 1
        return out

    def last_is_question(self) -> bool:
        h = [u for u in self.human() if u.text]
        return bool(h) and h[-1].text.rstrip().endswith("?") and h[-1] is self.items[-1]

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
    robot_turns_since_human: int = 0  # robot contributions nobody has answered yet
    last_is_question: bool = False    # the last human turn asked something (often of the robot)
    unattributed_s: float = 0.0       # speech we could not yet link to a face


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
    "max_unanswered": 3,        # robot contributions in a row with no reply -> close politely
}


def quiet_person(s: FloorState, R: dict) -> str | None:
    """Least-heard person whose speaking share is clearly below an equal split."""
    if len(s.present) < 2:
        return None
    total = sum(s.heard_s.get(p, 0.0) for p in s.present)
    if s.unattributed_s > total and total > 0:
        return None                     # most speech is not linked to a face yet: we don't know who is quiet
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
    if s.last_is_question and s.human_turns_since_robot >= 1 and s.since_robot_s >= 2.0:
        return FloorDecision("respond", reason="someone asked a question - answer it")
    if s.robot_turns_since_human >= R["max_unanswered"]:
        return FloorDecision("close", reason=f"{s.robot_turns_since_human} contributions without a reply")
    lull = R["lull_s"] * (1 + s.robot_turns_since_human)          # back off when nobody answers
    if s.silence_s >= lull and s.since_robot_s >= lull:
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
- Language: speak in the language given as "Reply language" (the language of the person you are \
answering, or the group's main language). Say each thing ONCE: never repeat the same content in both \
languages. Use simple Kiswahili and simple English.
- Stay on the topic the group is discussing NOW (see the last turns): if they are talking about one place \
or idea (e.g. Nairobi), build on THAT. Do not bring in new places or options unless your task says you may.
- If someone asks YOU something (e.g. to repeat the question, what you think, what you can see), answer \
that first, directly and briefly. To repeat the question, repeat the current discussion question.
- If the discussion has drifted from the topic, gently connect it back.
- Vary how you end: do NOT end every turn with a question, and never repeat the question you asked in your \
previous turn.
- If the participants clearly want to stop (e.g. "let's end here", "thank you, bye"), reply with kind \
"closing": thank them in one sentence.
- The transcript comes from automatic speech recognition and can contain errors; never quote \
garbled words, and if something is unclear, do not guess wildly.
- Do not comment on anyone's appearance unless they ask you about it; then answer from "What you can \
see" below, kindly and briefly. Never name or single out a person, never pressure anyone \
to speak, never mention that you are an AI model, never give personal data.

Reply with JSON only:
{"kind": "opinion" | "followup" | "summary" | "bridge" | "question" | "answer" | "closing",
 "segments": [{"lang": "en" | "sw", "text": "..."}]}
Use one segment per language run, in speaking order (e.g. a Kiswahili phrase then English)."""

ACTION_BRIEF = {
    "respond": "The group just finished a few turns. React to what was said about the place or idea they are "
               "discussing now: add a balanced view on it, connect two of their ideas, or ask a follow-up. "
               "Do not introduce new options.",
    "lull": "The group has gone quiet. Re-open the discussion about what they were discussing; you may suggest "
            "at most ONE new, concrete idea, then ask an easy open question.",
    "close": "Time is up. Summarise the main ideas you heard in at most two short sentences (max 50 words) "
             "and thank the group.",
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


def shorten(text: str, max_words: int) -> str:
    """Spoken output must stay short: cut at the last sentence end within max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > 10 else cut.rstrip(",;:") + "..."


def validate(obj: dict | None, max_chars: int = 400, max_words: int = 45) -> Contribution | None:
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
    budget, short = max_words, []
    for lang, text in out:                       # keep the word budget across segments
        if budget <= 3:
            break
        t = shorten(text, budget)
        short.append((lang, t))
        budget -= len(t.split())
    out = short
    kind = str(obj.get("kind", "opinion"))
    return Contribution(kind if kind in ("opinion", "followup", "summary", "bridge", "question", "answer",
                                         "closing") else "opinion",
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

    def previous_question(self) -> str | None:
        qs = self.topic.get("questions", [])
        return qs[self._q_i - 2] if 2 <= self._q_i <= len(qs) + 1 else None

    def next_question(self) -> str | None:
        qs = self.topic.get("questions", [])
        if self._q_i < len(qs):
            self._q_i += 1
            return qs[self._q_i - 1]
        return None

    def messages(self, action: str, transcript: Transcript, n_people: int, names: dict,
                 seen: dict | None = None, reply_lang: str = "en") -> list:
        qs = "\n".join(f"- {q}" for q in self.topic.get("questions", []))
        user = (f"Topic: {self.topic.get('title', '')}\n{self.topic.get('intro', '')}\n"
                f"Discussion questions:\n{qs}\n\n"
                f"Current question: {self.current_question() or '(none yet)'}\n"
                f"People present: {n_people}\n"
                f"What you can see (only mention if asked): "
                f"{'; '.join(f'{names.get(k, k)}: {v}' for k, v in (seen or {}).items()) or 'nothing'}\n\n"
                f"Recent discussion:\n{transcript.render(names=names)}\n\n"
                f"Languages used recently (turns): {transcript.language_mix() or 'none yet'}; "
                f"last speaker used: {transcript.last_lang()}\n"
                f"Reply language: {'Kiswahili' if reply_lang == 'sw' else 'English'}\n\n"
                f"Your task now: {ACTION_BRIEF[action]}")
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def current_question(self) -> str | None:
        qs = self.topic.get("questions", [])
        return qs[self._q_i - 1] if 0 < self._q_i <= len(qs) else None

    def compose(self, action: str, transcript: Transcript, n_people: int, names: dict | None = None,
                seen: dict | None = None, reply_lang: str | None = None) -> Contribution:
        """Always returns something sayable (model output or a scripted fallback)."""
        if self.chat is not None and self.chat.available:
            reply_lang = reply_lang or transcript.last_lang()
            reply = self.chat.chat(self.messages(action, transcript, n_people, names or {}, seen, reply_lang),
                                   deadline_s=self.deadline_s)
            c = validate(extract_json(reply), max_words=60 if action == "close" else 45)
            if c:
                c.segments = one_language(c.segments, reply_lang)
                c.source = getattr(self.chat, "used", None) or "model"
                return c
            self.last_error = "no reply / timeout" if reply is None else f"unusable reply: {reply[:120]!r}"
            log.warning("model contribution failed: %s", self.last_error)
        return self.fallback(action, transcript, reply_lang)

    def fallback(self, action: str, transcript: Transcript, reply_lang: str | None = None) -> Contribution:
        lang = reply_lang or transcript.last_lang()
        if action in ("lull", "opening"):
            q = self.next_question()
            if q:
                return Contribution("question", [("en", q)], "fallback")
        pool = self.fb.get(action) or self.fb.get("respond") or [{"en": "Interesting. Does anyone see it differently?"}]
        line = self.rng.choice(pool)
        segs = [(lang, line[lang])] if lang in line else [(k, v) for k, v in line.items() if k in ("sw", "en")][:1]
        return Contribution("bridge", segs, "fallback")


def one_language(segments: list, lang: str) -> list:
    """Never say the same thing twice in two languages: keep the reply-language
    segments; if the model used only the other language, keep what it said."""
    same = [(l, t) for l, t in segments if l == lang]
    return same or segments


# =============================================================================
# Requests the robot must understand reliably (rules, English + Kiswahili).
# Checked on every transcript before the model is involved.
# =============================================================================
INTENTS = [
    ("previous_question", r"previous question|last question|question before|go back to the (last|previous)|"
                          r"swali (lililopita|la awali|la mwanzo|lililotangulia)"),
    ("repeat_question", r"(repeat|say again|what (was|is|'s)) (the |that |your )?question|what was the question|"
                        r"rudia swali|swali (ni|lilikuwa|ulilouliza) (nini|gani)|swali gani"),
    ("next_question", r"next question|another question|(let'?s|we can) move on|"
                      r"swali (lingine|linalofuata|jingine)|tuendelee na swali"),
    ("end", r"\b(let'?s|we can|we should|i think we can) (end|stop|finish)|end (the|this|our) (conversation|discussion|chat)|"
            r"that'?s all( for now)?|\bgood ?bye\b|\bbye\b|kwaheri|tumalize|tumemaliza|tuishie hapa|inatosha|tusimame hapa"),
    ("repeat_last", r"did ?n[o']?t (hear|catch|get|understand) (you|that|it|what)|(can|could) you (repeat|say that again)|"
                    r"say (that|it) again|\bpardon\b|come again|sikusikia|sijasikia|sijaelewa|rudia|sema tena|"
                    r"unaweza kurudia"),
]


def detect_intent(text: str | None) -> str | None:
    """'repeat_question' | 'previous_question' | 'repeat_last' | 'next_question' | 'end' | None"""
    import re
    if not text:
        return None
    t = text.lower()
    for name, pattern in INTENTS:
        if re.search(pattern, t):
            return name
    return None


def contribution_log(c: Contribution) -> str:
    return json.dumps({"kind": c.kind, "segments": c.segments, "source": c.source}, ensure_ascii=False)
