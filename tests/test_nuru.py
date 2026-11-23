"""Unit + integration tests.  Run:  python -m pytest -q"""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from nuru.analysis import analyze, read_log, session_measures
from nuru.config import load_config
from nuru.controller import NuruController, auto_invite_rule, seats_from_users
from nuru.cues import lerp, nearest_neighbour, perform_invitation, pick_prompt
from nuru.design import schedule, williams_orders
from nuru.participation import Tracker, gini
from nuru.robot import MockRobot
from nuru.vad import MultiChannelVAD


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ------------------------------------------------------------------ Gini
def test_gini_extremes():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75)   # (n-1)/n for n = 4
    assert gini([0, 0, 0, 0]) == 0.0


# --------------------------------------------------------------- tracker
def test_tracker_time_turns_and_response():
    c = FakeClock()
    tr = Tracker(list("ABCD"), clock=c)
    tr.start("A"); c.t += 10
    tr.start("B"); c.t += 5
    tr.stop()
    assert tr.time_s["A"] == pytest.approx(10)
    assert tr.time_s["B"] == pytest.approx(5)
    assert tr.turns == {"A": 1, "B": 1, "C": 0, "D": 0}

    seat = tr.choose("least_active")
    assert seat == "C"                   # C and D tie at 0 -> first in order
    tr.register_invite(seat)
    c.t += 3; tr.open_response_window(seat)
    c.t += 2
    info = tr.start("C")
    assert info["response"] and info["latency_s"] == pytest.approx(2)
    m = tr.metrics()
    assert m["responses"]["C"] == 1 and m["response_rate"] == 1.0
    assert m["voluntary_turns"]["C"] == 0


def test_answer_during_prompt_counts_as_response():
    c = FakeClock()
    tr = Tracker(list("AB"), clock=c)
    tr.register_invite("B")
    c.t += 1
    assert tr.start("B")["latency_s"] == 0.0


def test_no_response_expires():
    c = FakeClock()
    tr = Tracker(list("AB"), clock=c, response_window_s=8)
    tr.register_invite("B"); tr.open_response_window("B")
    c.t += 9
    assert tr.expire_invitation().seat == "B"
    assert tr.start("B") == {}          # too late: a voluntary turn
    assert tr.voluntary["B"] == 1


def test_policies():
    tr = Tracker(list("ABCD"), clock=FakeClock(), no_repeat_window=2)
    assert [tr.choose("round_robin") for _ in range(5)] == list("ABCDA")
    # least_active never invites the same person twice in a row (no singling out)
    picks = []
    for _ in range(6):
        s = tr.choose("least_active"); tr.register_invite(s); picks.append(s)
    assert all(a != b for a, b in zip(picks, picks[1:]))


def test_auto_invite_rules():
    mod = {"auto_invite_silence_s": 7, "turn_end_gap_s": 1.5, "turn_end_invite_interval_s": 40,
           "min_gap_between_invites_s": 12}
    assert auto_invite_rule(mod, True, 20, 100) is None        # someone speaking
    assert auto_invite_rule(mod, False, 20, 5) is None         # too soon after last invite
    assert auto_invite_rule(mod, False, 8, 20) == "lull"
    assert auto_invite_rule(mod, False, 2, 20) is None
    assert auto_invite_rule(mod, False, 2, 45) == "turn_end"


# ------------------------------------------------------------------ cues
def _cfg():
    """Tests use a fixed 4-seat layout, independent of the room set-up in config/nuru.json."""
    cfg = load_config()
    cfg.seats = {"A": [-0.6, 0.0, 1.2], "B": [-0.2, 0.0, 1.2], "C": [0.2, 0.0, 1.2], "D": [0.6, 0.0, 1.2]}
    return cfg


def test_c3_cue_sequence_and_release():
    cfg = _cfg()
    robot = MockRobot(speed=0, quiet=True)
    prof = cfg.profile("c3")
    for k in ("group_glance_s", "pair_glance_s", "gaze_hold_s", "wait_s"):
        prof[k] = 0.01
    perform_invitation(robot, prof, "C", cfg.seats, cfg.group_centre, threading.Event())
    kinds = [a[1] for a in robot.actions]
    says = [a[3] for a in robot.actions if a[1] == "say"]
    assert kinds[0] == "attend"                                  # group glance first
    assert "gesture" in kinds                                    # soft nod
    assert says[0] in prof["prompts"]["sw"] and says[1] in prof["prompts"]["en"]   # bilingual sw -> en
    assert "Hakuna shida." in says                               # gentle release, never insists


def test_c1_c2_share_cue_form():
    cfg = _cfg()
    c1, c2 = cfg.profile("c1"), cfg.profile("c2")
    differ = {k for k in c1 if c1[k] != c2[k]} - {"label", "description"}
    assert differ == {"policy", "idle_gaze"}, "C1 vs C2 must differ only in targeting (adaptivity)"
    c3 = cfg.profile("c3")
    assert c3["policy"] == c2["policy"], "C2 vs C3 must share the participation model"


def test_invitation_stops_when_floor_taken():
    cfg = _cfg()
    robot = MockRobot(speed=0, quiet=True)
    ev = threading.Event()
    prof = cfg.profile("c3")
    threading.Timer(0.3, ev.set).start()
    t0 = time.time()
    assert perform_invitation(robot, prof, "A", cfg.seats, cfg.group_centre, ev) is True
    assert time.time() - t0 < 2.0
    assert not any(a[1] == "say" and a[3] == "Hakuna shida." for a in robot.actions)


def test_geometry_helpers():
    seats = {"A": (-0.6, 0, 1.2), "B": (-0.2, 0, 1.2), "C": (0.2, 0, 1.2)}
    assert nearest_neighbour("A", seats) == "B"
    assert lerp((0, 0, 0), (1, 2, 3), 0.5) == (0.5, 1.0, 1.5)
    users = [{"location": {"x": 0.5, "y": 0, "z": 1}}, {"location": {"x": -0.5, "y": 0, "z": 1}},
             {"location": {"x": 0.0, "y": 0, "z": 1}}]
    got = seats_from_users(users, seats)
    assert got["A"][0] == -0.5 and got["C"][0] == 0.5
    assert seats_from_users(users[:2], seats) is None


def test_prompt_translation_aligned():
    import random
    p = {"prompt_mode": "bilingual", "prompts": {"en": ["e0", "e1"], "sw": ["s0", "s1"]}}
    for seed in range(10):
        (l1, t1), (l2, t2) = pick_prompt(p, random.Random(seed))
        assert (l1, l2) == ("sw", "en") and t1[1] == t2[1]


# ---------------------------------------------------------------- design
def test_williams_design_balanced():
    orders = williams_orders(["c1", "c2", "c3"])
    assert len(orders) == 6
    # every condition appears in every position equally often
    for pos in range(3):
        assert sorted(o[pos] for o in orders) == ["c1", "c1", "c2", "c2", "c3", "c3"]
    # carry-over: every ordered pair (x -> y) occurs equally often
    pairs = [(o[i], o[i + 1]) for o in orders for i in range(2)]
    assert len(set(pairs)) == 6 and all(pairs.count(p) == 2 for p in set(pairs))
    # topic x condition balance over 6 groups
    combos = [(b["condition"], b["task"]) for g in schedule(6) for b in g["blocks"]]
    assert all(combos.count(c) == 2 for c in set(combos))


# ------------------------------------------------------------------- VAD
def test_vad_detects_speaker_and_suppresses_crosstalk():
    starts, stops = [], []
    vad = MultiChannelVAD(list("ABCD"), on_start=starts.append, on_stop=stops.append)
    rng = np.random.default_rng(0)
    n = 480   # 30 ms @ 16 kHz
    noise = lambda: rng.normal(0, 0.001, (n, 4))
    for _ in range(30):                      # 0.9 s silence to learn the noise floor
        vad.process(noise())
    for _ in range(40):                      # 1.2 s: B speaks loudly, A picks up crosstalk
        f = noise()
        f[:, 1] += rng.normal(0, 0.1, n)     # B: +40 dB
        f[:, 0] += rng.normal(0, 0.01, n)    # A: +20 dB crosstalk
        vad.process(f)
    for _ in range(40):                      # 1.2 s silence
        vad.process(noise())
    assert starts == ["B"] and stops == ["B"]


# --------------------------------------------------- end-to-end (mock robot)
def test_controller_end_to_end(tmp_path):
    cfg = _cfg()
    cfg.log_dir = tmp_path
    for c in ("c1", "c2", "c3"):
        for k in ("group_glance_s", "pair_glance_s", "gaze_hold_s", "wait_s"):
            cfg.profiles[c][k] = 0.05
    robot = MockRobot(speed=0, quiet=True)
    ctrl = NuruController(cfg, robot, group="T01")
    ctrl.set_meta({"D": "quiet"})
    ctrl.start_condition("c2", "T1", intro=False)
    ctrl.speak("A"); time.sleep(0.3)
    ctrl.speak("B"); time.sleep(0.2)
    ctrl.silence()
    seat = ctrl.invite()
    assert seat == "C"
    time.sleep(0.15)
    ctrl.speak("C"); time.sleep(0.2)          # invitee answers
    ctrl.silence()
    time.sleep(0.3)
    m = ctrl.end_condition()
    ctrl.shutdown()
    assert m["responses"]["C"] == 1 and m["invitations"]["C"] == 1

    rows = read_log(ctrl.log.path)
    blocks = session_measures(rows)
    assert len(blocks) == 1 and blocks[0]["condition"] == "c2"
    assert blocks[0]["response_rate"] == 1.0 and blocks[0]["quiet_seats"] == "D"
    assert blocks[0]["gini_time"] == pytest.approx(m["gini_time"], abs=0.02)

    res = analyze(tmp_path, tmp_path / "out", plot=False)
    assert Path(res["report"]).exists()


# ------------------------------------------------------------- live mode
from nuru.live import (Memory, LiveSession, play_scenario, plan_after_listen, plan_between_questions,
                       plan_engagement, plan_invitee, plan_mode)
from nuru.perception import PeopleTracker, Percept, Person


def _pc(*people, t=100.0):
    return Percept(t, [Person(pid, (x, 0.0, z), first, t) for pid, x, z, first in people])


def test_perception_confirms_and_loses_people():
    c = FakeClock()
    tr = PeopleTracker(MockRobot(quiet=True), poll_s=0.3, confirm_s=1.0, lost_after_s=3.0, clock=c)
    face = {"id": "u1", "location": {"x": 0.1, "y": 0.0, "z": 1.0}}
    far = {"id": "u2", "location": {"x": 0.0, "y": 0.0, "z": 4.0}}
    tr.update([face, far]); assert tr.snapshot().n == 0          # not yet confirmed; far face ignored
    c.t += 1.1; tr.update([face])
    p = tr.snapshot(); assert [x.id for x in p.people] == ["u1"] and p.arrived == ["u1"]
    c.t += 1.0; tr.update(None)                                   # camera hiccup: keep last state
    assert tr.snapshot().n == 1
    c.t += 3.5; tr.update([])
    p = tr.snapshot(); assert p.n == 0 and p.left == ["u1"]


def test_plan_social_feature_changes_mode():
    mem = Memory()
    assert plan_engagement(_pc(), mem, 1.6).action == "idle"
    assert plan_engagement(_pc(("a", 0, 2.2, 90)), mem, 1.6).action == "beckon"
    solo = plan_engagement(_pc(("a", 0, 1.0, 90)), mem, 1.6)
    group = plan_engagement(_pc(("a", -0.4, 1.0, 90), ("b", 0.4, 1.0, 95)), mem, 1.6)
    assert (solo.action, solo.mode) == ("start", "solo")
    assert (group.action, group.mode) == ("start", "group")


def test_plan_newcomer_and_departure():
    mem = Memory(greeted={"a"})
    d = plan_between_questions(_pc(("a", 0, 1.0, 50), ("b", 0.5, 1.0, 98)), mem)
    assert (d.action, d.target, d.mode) == ("welcome", "b", "group")
    assert plan_between_questions(_pc(), mem).action == "farewell"


def test_plan_invitee_least_heard_then_longest_present():
    p = _pc(("a", -0.4, 1.0, 10), ("b", 0.0, 1.0, 50), ("c", 0.4, 1.0, 90))
    mem = Memory(heard_s={"a": 5.0})
    assert plan_invitee(p, mem) == "b"            # b and c unheard; b has been there longer
    mem.invited.append("b")
    assert plan_invitee(p, mem) == "c"            # never the same person twice in a row


def test_plan_failure_handling():
    assert plan_after_listen("speech", "solo", 0, True).action == "acknowledge"
    assert plan_after_listen("silence", "solo", 0, True).action == "repair"
    assert plan_after_listen("silence", "solo", 1, True).action == "release"
    assert plan_after_listen("silence", "group", 0, True).action == "release"   # never insists
    assert plan_after_listen("silence", "group", 0, False).action == "invite"
    assert plan_after_listen("error", "solo", 0, True).action == "repair_tech"
    assert plan_after_listen("error", "solo", 1, True).action == "repair_tech_move_on"


@pytest.mark.parametrize("scenario,expect", [("solo", "Hello, I am Nuru"), ("group", "Hakuna shida.")])
def test_live_session_runs_without_operator(tmp_path, scenario, expect):
    cfg = _cfg(); cfg.log_dir = tmp_path
    cfg.live["interaction"] = "questions"      # the Lab 1 flow; never touch real audio/camera feeds in tests
    robot = MockRobot(speed=0.02, quiet=True)
    sess = LiveSession(cfg, robot, verbose=False, speed=0.02,
                       sense=PeopleTracker(robot, poll_s=0.05, confirm_s=0.1, lost_after_s=0.5))
    play_scenario(robot, scenario, 0.02, sess.stop_event)
    t = threading.Thread(target=sess.run, kwargs={"max_interactions": 1}, daemon=True)
    t.start(); t.join(20)
    assert not t.is_alive(), "live session did not finish"
    says = " ".join(a[3] for a in robot.actions if a[1] == "say")
    assert expect in says
    assert any(a[1] == "led" for a in robot.actions)              # state feedback on the LED
