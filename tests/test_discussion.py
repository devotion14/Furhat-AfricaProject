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


def test_camera_hint_links_voice():
    l = SpeakerLinker()
    assert l.link("V1", ["p1", "p2"], seen="p1") == ("p1", "camera: mouth moving")
    assert l.link("V1", ["p1", "p2"], seen="p2")[0] == "p1"             # a known voice wins over the camera
    assert l.link(None, ["p1", "p2"], seen="p2")[0] == "p2"


def test_camera_seat_timeline():
    from nuru.speaker_cam import CameraSpeakerHint
    h = CameraSpeakerHint.__new__(CameraSpeakerHint)
    import collections, threading
    h.events, h._lock = collections.deque([(0.0, None), (1.0, "A"), (4.0, None), (5.0, "B")]), threading.Lock()
    assert h.seat_during(1.0, 4.5) == "A"
    assert h.seat_during(3.5, 6.0) is None or h.seat_during(3.5, 6.0) == "B"
    assert h.seat_during(0.0, 0.5) is None


def test_appearance_validation_and_line():
    from nuru.appearance import clean, greeting_line
    r = clean({"people": [{"expression": "Serious", "top_colour": "Blue", "clothing": "school uniform"},
                          {"expression": "angry??", "top_colour": "red", "clothing": "t-shirt"}]}, 2)
    assert r[0] == {"expression": "serious", "colour": "blue", "clothing": "school uniform"}
    assert r[1]["expression"] == "calm"                                  # unknown -> neutral
    assert clean({"people": [{"expression": "calm"}]}, 2) is None        # wrong number of people
    assert greeting_line(r[0]).startswith("Hello to you in the blue school uniform.")


def test_whisper_hallucination_filter():
    from nuru.llm import speech_text
    assert speech_text({"text": " you", "segments": [{"text": " you", "no_speech_prob": 0.81}]}) == ""
    assert speech_text({"text": " .", "segments": [{"text": " .", "no_speech_prob": 0.1}]}) == ""
    assert speech_text({"text": "Habari", "segments": [{"text": " Habari", "no_speech_prob": 0.05}]}) == "Habari"


def test_long_model_output_is_shortened():
    from nuru.discussion import shorten
    long = "First sentence is here. " + "word " * 60
    assert shorten(long, 45) == "First sentence is here."
    c = validate({"segments": [{"lang": "en", "text": "One two three. " * 20}]}, max_words=10)
    assert len(c.text.split()) <= 10


def test_race_prefers_first_model_but_does_not_wait_forever():
    import time as _t
    from nuru.llm import FallbackChat

    class Fake:
        def __init__(self, name, delay, out):
            self.name, self.delay, self.out, self.timeout_s, self.available = name, delay, out, 5, True

        def chat(self, m, temperature=0.7, timeout_s=None):
            _t.sleep(self.delay)
            return self.out
    fc = FallbackChat([Fake("minimax", 0.1, "A"), Fake("groq", 0.05, "B")], prefer_wait_s=0.5)
    assert fc.chat([], deadline_s=2) == "A" and fc.used == "minimax"
    fc = FallbackChat([Fake("minimax", 1.5, "A"), Fake("groq", 0.05, "B")], prefer_wait_s=0.3)
    t0 = _t.time()
    assert fc.chat([], deadline_s=2) == "B" and _t.time() - t0 < 0.6
    fc = FallbackChat([Fake("minimax", 0.05, None), Fake("groq", 0.05, None)], prefer_wait_s=0.3)
    assert fc.chat([], deadline_s=1) is None


def test_transcript_filters_and_direct_questions():
    from nuru.llm import norm_lang, plausible
    assert norm_lang("swahili (macrolanguage)") == "sw" and norm_lang("English") == "en"
    assert not plausible("ご視聴ありがとうございました。", "japanese", 3.4)
    assert not plausible("Thank you.", "en", 1.4)
    assert plausible("Thank you for giving us the options, I think we can end here.", "en", 6.2)
    assert not plausible("A classroom discussion in English and Kiswahili.", "en", 2.0)
    s = state(last_is_question=True, human_turns_since_robot=1, since_robot_s=3)
    assert plan_floor(s).action == "respond"
    assert plan_floor(state(last_is_question=True, human_turns_since_robot=1, someone_speaking=True)).action == "wait"


def test_short_clips_do_not_create_new_voices():
    b = VoiceBank(threshold=0.95)
    a, c = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    assert b.identify(a)[0] == "V1"
    assert b.identify(c, allow_new=False)[0] == "V1"          # too short to be a new person
    assert b.identify(c, allow_new=True)[0] == "V2"
    b.max_voices = 2
    assert b.identify(np.array([0, 0, 1.0]))[0] in ("V1", "V2")


def test_intents_english_and_kiswahili():
    from nuru.discussion import detect_intent
    cases = {"Can you repeat the question?": "repeat_question", "Rudia swali tafadhali": "repeat_question",
             "What was the previous question?": "previous_question", "Swali lililopita lilikuwa gani?": "previous_question",
             "Sorry, I didn't hear you": "repeat_last", "Sikusikia vizuri": "repeat_last",
             "Next question please": "next_question", "Tuendelee na swali lingine": "next_question",
             "I think we can end the conversation here.": "end", "Haya, kwaheri!": "end",
             "We should go to Nairobi this weekend.": None, "I like the museum.": None}
    for text, want in cases.items():
        assert detect_intent(text) == want, text


def test_one_language_per_turn():
    from nuru.discussion import one_language
    segs = [("sw", "Ni wazo zuri."), ("en", "It is a good idea.")]
    assert one_language(segs, "sw") == [("sw", "Ni wazo zuri.")]
    assert one_language([("en", "Only English.")], "sw") == [("en", "Only English.")]


def test_face_memory_recognises_returning_people():
    from nuru.faces import FaceMemory
    m = FaceMemory(threshold=0.9)
    a, b = np.array([1.0, 0.1, 0]), np.array([0, 1.0, 0.1])
    assert m.assign("user-0", a, ["user-1"]) == ("new", None)
    assert m.assign("user-1", b, ["user-0"]) == ("new", None)
    assert m.assign("user-0", a, ["user-1"])[0] == "known"
    # user-0 leaves; comes back as user-5
    assert m.assign("user-5", a * 1.02, ["user-1"]) == ("returning", "user-0")
    # a stranger is new, and someone present cannot be matched
    assert m.assign("user-6", np.array([0, 0, 1.0]), ["user-5", "user-1"]) == ("new", None)
    assert m.assign("user-7", b, ["user-1"]) == ("new", None)


def test_c1_invites_in_turn_c2_c3_invite_the_quiet():
    rr = {"policy": "round_robin"}
    s = state(heard_s={"p1": 2, "p2": 40}, next_in_turn="p2")
    d = plan_floor(s, rr)
    assert (d.action, d.target) == ("invite", "p2")              # C1 ignores who is quiet
    assert plan_floor(state(heard_s={"p1": 2, "p2": 40}, next_in_turn="p2")).target == "p1"   # C2/C3


def test_ready_during_break():
    from nuru.discussion import detect_ready
    assert detect_ready("We're ready") and detect_ready("Tuko tayari") and detect_ready("tuendelee")
    assert not detect_ready("I think the museum is better")
