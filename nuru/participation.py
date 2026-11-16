"""Participation model: who has spoken, for how long, and who to invite next.

The model is fed by speech events (from the Wizard-of-Oz panel or from the
per-participant voice-activity detector). It is the same in all three
conditions; only the *policy* used to pick an invitee and the *cues* used to
express the invitation differ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable


def gini(values) -> float:
    """Gini coefficient. 0 = perfectly equal, (n-1)/n = one person did everything."""
    vals = sorted(float(v) for v in values)
    n, total = len(vals), sum(vals)
    if n == 0 or total == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return (2 * cum) / (n * total) - (n + 1) / n


@dataclass
class Invitation:
    seat: str
    t_prompt_end: float | None    # when the robot finished the verbal prompt (None = still speaking)
    window_s: float
    responded: bool = False
    latency_s: float | None = None


@dataclass
class Tracker:
    seats: list[str]
    clock: Callable[[], float] = time.time
    no_repeat_window: int = 2
    response_window_s: float = 8.0

    time_s: dict = field(init=False)
    turns: dict = field(init=False)
    voluntary: dict = field(init=False)
    invited_n: dict = field(init=False)
    responses: dict = field(init=False)
    latencies: dict = field(init=False)
    current: str | None = field(init=False, default=None)
    started: float | None = field(init=False, default=None)
    last_speech_end: float = field(init=False)
    history: list = field(init=False)
    pending: Invitation | None = field(init=False, default=None)
    rr_index: int = field(init=False, default=0)

    def __post_init__(self):
        self.reset()

    def reset(self):
        z = lambda v=0: {s: v for s in self.seats}
        self.time_s, self.turns, self.voluntary = z(0.0), z(), z()
        self.invited_n, self.responses = z(), z()
        self.latencies = {s: [] for s in self.seats}
        self.current, self.started, self.pending = None, None, None
        self.last_speech_end = self.clock()
        self.history = []   # seats invited, in order
        self.rr_index = 0

    # -- speech events -------------------------------------------------------
    def start(self, seat: str) -> dict:
        """A participant started speaking. Returns info about invitation response."""
        if seat == self.current:
            return {}
        self.stop()
        now = self.clock()
        self.current, self.started = seat, now
        self.turns[seat] += 1
        info = {}
        inv = self.pending
        if inv and not inv.responded and seat == inv.seat:
            # Answering before the prompt has finished counts as latency 0.
            latency = 0.0 if inv.t_prompt_end is None else now - inv.t_prompt_end
            if latency <= inv.window_s:
                inv.responded = True
                inv.latency_s = max(0.0, latency)
                self.responses[seat] += 1
                self.latencies[seat].append(inv.latency_s)
                info = {"response": True, "seat": seat, "latency_s": inv.latency_s}
        if not info:
            self.voluntary[seat] += 1
        return info

    def stop(self):
        if self.current is not None:
            now = self.clock()
            self.time_s[self.current] += now - self.started
            self.last_speech_end = now
        self.current, self.started = None, None

    def live_time(self) -> dict:
        """Speaking time including the turn in progress."""
        t = dict(self.time_s)
        if self.current is not None:
            t[self.current] += self.clock() - self.started
        return t

    def silence_s(self) -> float:
        return 0.0 if self.current else self.clock() - self.last_speech_end

    # -- invitations ---------------------------------------------------------
    def choose(self, policy: str) -> str:
        if policy == "round_robin":
            seat = self.seats[self.rr_index % len(self.seats)]
            self.rr_index += 1
            return seat
        if policy == "least_active":
            t = self.live_time()
            recent = set(self.history[-self.no_repeat_window:]) if self.no_repeat_window else set()
            candidates = [s for s in self.seats if s not in recent and s != self.current] or self.seats
            # least speaking time; ties -> fewest turns -> fewest invitations
            return min(candidates, key=lambda s: (round(t[s], 1), self.turns[s], self.invited_n[s]))
        raise ValueError(f"unknown policy {policy!r}")

    def least_active(self) -> str:
        t = self.live_time()
        return min(self.seats, key=lambda s: (round(t[s], 1), self.turns[s]))

    def register_invite(self, seat: str):
        self.history.append(seat)
        self.invited_n[seat] += 1
        self.pending = Invitation(seat, None, self.response_window_s)

    def open_response_window(self, seat: str) -> Invitation:
        """Called when the verbal prompt ends: the response window starts now."""
        if not (self.pending and self.pending.seat == seat):
            self.pending = Invitation(seat, None, self.response_window_s)
        if self.pending.t_prompt_end is None:
            self.pending.t_prompt_end = self.clock()
        return self.pending

    def expire_invitation(self) -> Invitation | None:
        """Close the pending invitation if its window has passed. Returns it if unanswered."""
        inv = self.pending
        if (inv and not inv.responded and inv.t_prompt_end is not None
                and self.clock() - inv.t_prompt_end > inv.window_s):
            self.pending = None
            return inv
        if inv and inv.responded:
            self.pending = None
        return None

    # -- metrics -------------------------------------------------------------
    def metrics(self) -> dict:
        t = self.live_time()
        total_t, total_turns = sum(t.values()), sum(self.turns.values())
        least = min(self.seats, key=lambda s: t[s])
        inv_total = sum(self.invited_n.values())
        lat = [x for v in self.latencies.values() for x in v]
        return {
            "speaking_time_s": {s: round(t[s], 2) for s in self.seats},
            "turns": dict(self.turns),
            "voluntary_turns": dict(self.voluntary),
            "invitations": dict(self.invited_n),
            "responses": dict(self.responses),
            "gini_time": round(gini(t.values()), 4),
            "gini_turns": round(gini(self.turns.values()), 4),
            "least_active_seat": least,
            "least_active_share_time": round(t[least] / total_t, 4) if total_t else 0.0,
            "least_active_share_turns": round(min(self.turns.values()) / total_turns, 4) if total_turns else 0.0,
            "response_rate": round(sum(self.responses.values()) / inv_total, 4) if inv_total else None,
            "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else None,
            "total_time_s": round(total_t, 2),
            "total_turns": total_turns,
        }
