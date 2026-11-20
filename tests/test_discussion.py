"""Discussion mode: floor rules, model-output validation, fallbacks, voice linking."""
import random

import numpy as np

from nuru.discussion import Brain, FloorState, RobotTurn, Transcript, plan_floor, validate
from nuru.listener import Utterance
from nuru.llm import extract_json, strip_reasoning
from nuru.voices import SpeakerLinker, VoiceBank


def state(**kw):
    base = dict(someone_speaking=False, robot_speaking=False, silence_s=2.0, since_robot_s=30.0,
                since_invite_s=99.0, elapsed_s=60.0, human_turns_since_robot=0,
                present=["p1", "p2"], dwell_s={"p1": 60, "p2": 60}, heard_s={"p1": 20, "p2": 18})
    base.update(kw)
    return FloorState(**base)


def test_never_interrupts():
    assert plan_floor(state(someone_speaking=True, silence_s=0, human_turns_since_robot=9)).action == "wait"
    assert plan_floor(state(silence_s=0.8, human_turns_since_robot=9)).action == "wait"


def test_invites_quiet_person_and_not_twice():
    s = state(heard_s={"p1": 40, "p2": 2})
    d = plan_floor(s)
    assert (d.action, d.target) == ("invite", "p2")
    assert plan_floor(state(heard_s={"p1": 40, "p2": 2}, last_invited="p2")).action != "invite"
    assert plan_floor(state(heard_s={"p1": 40, "p2": 2}, since_invite_s=5)).action != "invite"


def test_lull_respond_close():
    assert plan_floor(state(silence_s=7, since_robot_s=10)).action == "lull"
    assert plan_floor(state(human_turns_since_robot=3)).action == "respond"
    assert plan_floor(state(human_turns_since_robot=3, since_robot_s=5)).action == "wait"
    assert plan_floor(state(elapsed_s=999)).action == "close"


def test_validate_model_output():
    c = validate(extract_json('```json\n{"kind":"opinion","segments":[{"lang":"sw","text":"Ni wazo zuri."},'
                              '{"lang":"en","text":"But cost matters too."}]}\n```'))
    assert c.segments == [("sw", "Ni wazo zuri."), ("en", "But cost matters too.")]
    assert validate({"segments": [{"lang": "fr", "text": "bonjour"}]}) is None
    assert validate({"segments": [{"lang": "en", "text": "x" * 500}]}) is None
    assert validate(None) is None
    assert strip_reasoning("<think>plan</think>{\"a\":1}") == '{"a":1}'


def test_brain_falls_back_without_model():
    t = Transcript()
    t.add(Utterance(0, 3, "Tunapaswa kwenda Mombasa", "sw", "V1", "p1"))
    b = Brain(None, {"questions": ["Q1?", "Q2?"]}, {"respond": [{"en": "Interesting.", "sw": "Inavutia."}]},
              rng=random.Random(0))
    assert b.compose("lull", t, 2).segments == [("en", "Q1?")]
    assert b.compose("respond", t, 2).segments == [("sw", "Inavutia.")]      # follows the group's language


def test_transcript_bookkeeping():
    t = Transcript()
    t.add(Utterance(0, 4, "a", "en", "V1", "p1"))
    t.add(RobotTurn("ok", "bridge"))
    t.add(Utterance(5, 7, "b", "sw", "V2", "p2"))
    t.add(Utterance(8, 9, None, None, "V2", "p2"))
    assert t.human_turns_since_robot() == 2
    assert t.heard_s() == {"p1": 4, "p2": 3}
    assert t.last_lang() == "sw"
    assert "words not recognised" in t.render()


def test_voice_bank_and_linker():
    b = VoiceBank(threshold=0.9)
    a, c = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert [b.identify(v)[0] for v in (a, c, a * 1.01)] == ["V1", "V2", "V1"]
    l = SpeakerLinker()
    assert l.link("V1", ["p1", "p2"]) == (None, "not yet linked to a face")
    assert l.link("V1", ["p1", "p2"], invited="p2")[0] == "p2"
    assert l.link("V2", ["p1", "p2"])[0] == "p1"


def test_backs_off_and_closes_when_nobody_answers():
    assert plan_floor(state(silence_s=7, since_robot_s=7, robot_turns_since_human=1)).action == "wait"
    assert plan_floor(state(silence_s=13, since_robot_s=13, robot_turns_since_human=1)).action == "lull"
    assert plan_floor(state(robot_turns_since_human=3)).action == "close"


def test_no_targeted_invite_while_speakers_unknown():
    s = state(heard_s={"p1": 5, "p2": 0}, unattributed_s=30)
    assert plan_floor(s).action != "invite"
