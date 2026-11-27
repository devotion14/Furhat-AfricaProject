# Nuru (Light) - Furhat deployment

**Culturally adapted non-verbal behaviours for equitable participation in high-ratio classrooms.**

Luke Tsalwa · Celestino M.K. Kariuki · Devotha R. Lyakurwa 

KTH Royal Institute of Technology

Nuru is a Furhat robot that moderates small-group discussions. It keeps track of how much each
learner has spoken and invites quieter learners to contribute. The project tests whether the robot's
gaze, head and turn-taking cues work better when they are **co-designed with East African students
and teachers** than when they follow Western defaults.

This repository is the complete working system for the study:

| You want to… | Command | Covers |
|---|---|---|
| Check the robot is ready | `python -m nuru check` | setup |
| **Lab 1 demo: fully autonomous interaction** | `python -m nuru live` (rehearse: `--mock --scenario group`) | DD2413 Lab 1, see [docs/LAB1_GRADING_GUIDE.md](docs/LAB1_GRADING_GUIDE.md) |
| Run a study session | `python -m nuru panel`, then open http://localhost:5000 | Phase 3, Study A |
| Run a fully automatic session | `python -m nuru session --group G01` | Phase 3, Study A |
| Have the robot greet and comment on each person | `python -m nuru describe` | social-features demo |
| Calibrate camera-based speaker detection | `python -m nuru.avsd calibrate --furhat <robot-ip>` | setup |
| See C1 / C2 / C3 side by side | `python -m nuru demo` | demo |
| Record co-design clips | `python -m nuru clips --list` | Phase 1 |
| Aim the robot's gaze at the seats | `python -m nuru calibrate` | setup |
| Get the group schedule | `python -m nuru plan --groups 10` | Phase 3 design |
| Turn logs into results | `python -m nuru analyze logs/` | Phase 3 measures |
| Test the pipeline / power analysis | `python -m nuru simulate --power` | planning |

Add `--mock` to any robot command to run it without a robot.

---

## Quick start (5 minutes)

```powershell
# 1. Install (once)
pip install -r requirements.txt
#    Only for camera-based speaker detection (session, describe, panel --avsd), Python 3.10-3.12:
pip install mediapipe opencv-python pyzmq

# 2. On the robot: start the "Remote API" skill (Furhat web UI -> Skills -> Remote API -> Start)
#    The robot's IP is in config/nuru.json (currently 192.168.1.110)

# 3. Test every capability Nuru uses
python -m nuru check

# 4. See the three conditions on the robot (about 1 minute)
python -m nuru demo

# 5. Open the operator panel and run a session
python -m nuru panel
#    -> open http://localhost:5000 in a browser
```

Expected output of `check`:

```
Checking Furhat at 192.168.1.110:54321
  [OK ] Remote API reachable
  [OK ] voice en = AsiliaNeural
  [OK ] voice sw = ZuriNeural
  [OK ] built-in Nod gesture
  [OK ] custom gesture NuruSoftNod
  [OK ] gaze to group centre
  [OK ] speech (en)
  [OK ] speech (sw)
All good - ready for a session.
```

---

## Live discussion mode (`python -m nuru live`)

The robot runs a whole small-group discussion by itself, in English and Kiswahili:

1. **Notices people approaching** (Furhat face tracker), beckons from a distance and greets the group,
   with short natural pauses between steps.
2. **Greets each person**: expression and clothing (colour and type) from the camera ("Hello to you in
   the blue hoodie…"). Clothing type comes from MiniMax vision; if that is slow, expression and colour come
   from the laptop's own camera analysis.
3. **Listens without interrupting**: it transcribes speech (Groq Whisper, English and Kiswahili) and
   tells **voices apart**. Each voice is linked to a face by mouth movement, by who answered an
   invitation, or by elimination.
4. **Recognises people who come back** within the session (face signatures, kept in memory only):
   "Karibu tena!". It **counts people as they come and go** and says the number when it changes.
5. **Takes part**: fixed rules decide *when* it may speak (only after a pause). Groq (with MiniMax as
   backup) decides *what* to say: balanced views that stay on what the group is discussing. Each turn is
   in **one language**, the one used by the person it is answering.
6. **Understands requests** in English and Kiswahili: repeat the question, previous question, "I didn't
   hear you", next question, let's end.
7. **Body language**: it looks at whoever is talking and nods now and then. After 30 s it glances at the
   quiet person. It invites quiet people with the C3 cues from the paper (bilingual and never insisting).
   **LED**: green when someone is talking, pink when the robot talks, blue when it is thinking.
8. **Never stalls**: timeouts, fallbacks and scripted lines throughout. It backs off when nobody answers
   and closes politely.

```powershell
# once: API keys (never committed)
copy config\keys.example.json config\keys.json     # then paste your MiniMax and Groq keys
# on the robot web UI: Settings -> External feeds -> enable Camera feed and Audio feed

python -m nuru live                                   # the real thing
python -m nuru live --mock --scenario discussion --speed 0.2   # rehearse without robot or keys
python -m nuru live --questions                       # the Lab 1 flow (ask / listen / acknowledge)
```

All settings (rules, timings, models, languages, whether appearance is mentioned) are in
[config/live.json](config/live.json). Code: [nuru/live.py](nuru/live.py) (loop),
[nuru/discussion.py](nuru/discussion.py) (when and what to say), [nuru/listener.py](nuru/listener.py)
(listening), [nuru/voices.py](nuru/voices.py) (voices), [nuru/appearance.py](nuru/appearance.py),
[nuru/llm.py](nuru/llm.py) (cloud models).

**What leaves the laptop:** audio clips of each utterance (to Groq for transcription), one camera snapshot
per group (to Groq), and the recent transcript text (to MiniMax/Groq). Voiceprints stay on the laptop and
are discarded after each interaction. Tell participants this before the session.

---

## Running a group test (3 sessions: C1 -> C2 -> C3)

`python -m nuru live` runs the whole test by itself. Nobody needs to touch the laptop.

```powershell
python -m nuru live --group G01                    # automatic 15 s breaks
python -m nuru live --group G01 --questionnaire    # breaks wait for the questionnaire
```

| Step | What the robot does | Length |
|---|---|---|
| Arrival | Notices people, greets the group ("Habari zenu! Karibuni…"), then greets each person by clothing and expression | ~30 s |
| **Session 1: C1, topic T1** | Discussion. Invites the **next person in turn** with **plain English cues** (looks at them for 3 s, "What do you think?") | 5 min |
| Break | "Let's take a short break…" | 15 s, or until the questionnaire is done |
| **Session 2: C2, topic T2** | "Welcome back! Let's start the next topic." Invites the **quietest person** with the **same plain cues** | 5 min |
| Break | as above | |
| **Session 3: C3, topic T3** | Invites the **quietest person** with the **Nuru cues**: glance at the group, then the pair, a short look with a soft nod and smile, a bilingual prompt ("Je, una maoni? What do you think?"), looks back at the group, never insists ("Hakuna shida") | 5 min |
| Closing | Summary and thanks | |

**What stays the same in all three sessions:**
- It listens without interrupting and contributes balanced opinions in the language being used.
- It answers questions put to it, and looks at and nods to whoever is speaking.
- It understands "repeat the question", "I didn't hear", "next question", "let's end" (which ends the current session) and "we're ready" (which ends a break).

The only differences are the ones the study tests:
- **C1 vs C2: adaptivity.** Who is invited: the next in turn, or the quietest.
- **C2 vs C3: cultural adaptation.** How they are invited: plain cues, or Nuru cues.

In C2 and C3 the robot also glances briefly at the quietest person after 30 s on the active speakers.

**Topics** ([config/tasks.json](config/tasks.json)):

| Topic | Introduction | Questions the robot can ask |
|---|---|---|
| T1 School water project | Your school has received money for one project to improve access to clean water. | Which project, and why? · Who in the community should be involved? · What problems might it face? · How would you know after a year that it worked? |
| T2 Planning a class trip | Your class can go on a one-day trip anywhere within your country. | Where, and why? · What should students learn? · How can it be affordable for every family? · What rules should the class agree on? |
| T3 A better school library | Your school wants more students to use its library. | What would make students visit more? · Which books or materials first? · Books in local languages: why or why not? · How could older students help younger ones read? |

**Breaks:**
- **Automatic (default):** 15 s. Saying "we're ready" / "tuko tayari" or pressing Enter skips the break.
- **`--questionnaire`:** "Please fill in the questionnaire, and tell me when you are ready." The robot waits until someone says "we're ready" / "tuko tayari" or the operator presses Enter (at most 15 min). Use this mode for real study groups.
- **If everyone leaves:** the robot says it will wait for one minute. It continues if they come back, and ends otherwise.

**Other options:** `--conditions c2 c3 c1` (a different order, e.g. from `python -m nuru plan`), `--minutes 3`.

### For the questionnaires
- **Give one post-session questionnaire after each session (C1, C2, C3), and one final questionnaire at the end.** Use `--questionnaire` so the robot waits.
- **Label each sheet with the group id (e.g. G01) and the condition (c1/c2/c3)**, so the answers can be matched to the log.
- **The existing Likert items (1–7)** from [docs/STUDY_PROTOCOL.md](docs/STUDY_PROTOCOL.md) fit these sessions:

  | Code | Statement |
  |---|---|
  | `fair_attention` | The robot shared its attention fairly among us. |
  | `felt_invited` | I felt invited to share my ideas. |
  | `felt_pressured` | I felt pressured to speak. |
  | `singled_out` | I felt singled out by the robot. |
  | `appropriate` | The robot's behaviour was appropriate for a classroom I know. |
  | `comfortable` | I felt comfortable with how the robot looked at me. |
  | `respectful` | The robot was respectful. |
- **Behaviours you may also want to ask about** (they are the same in all sessions, so they belong in the final questionnaire):
  - being greeted by clothing and expression;
  - the robot's own opinions;
  - its English/Kiswahili switching;
  - the LED colours (green = someone talking, pink = robot talking, blue = thinking, amber = break);
  - being recognised after leaving and coming back;
  - whether it ever interrupted.
- **Entering the answers:** copy `data/questionnaire_template.csv` to `data/questionnaire.csv`, with one row per person, condition and item. Then `python -m nuru analyze logs/ --questionnaire data/questionnaire.csv` adds the answers to the report.
- **What the log measures automatically (per session):**
  - speaking time and turns per person, and how equal they are (Gini);
  - the share of the least active person;
  - invitations, the response rate, the response latency and voluntary turns;
  - for the full study, the C1 vs C2 and C2 vs C3 comparisons.

  Run it with `python -m nuru analyze logs/`.
- **What leaves the laptop (tell participants):**
  - audio clips of each utterance (Groq, for transcription);
  - camera snapshots (MiniMax, for clothing);
  - the transcript text (Groq/MiniMax, for the robot's replies).

  Face and voice signatures stay on the laptop, in memory only, and are deleted when the program stops.

---

## Automatic sessions and speaker detection

Nuru can work out by itself who is speaking, using **Furhat's own camera and microphone**. No extra
hardware is needed. The microphone tells it *that* someone is talking, and the camera tells it *who*: the
face whose mouth is moving most. Faces are matched to seats by where they appear in the image.

```powershell
# Once per room setup: everyone sits down, about 5 s (writes config/avsd.json)
python -m nuru.avsd calibrate --furhat 192.168.1.110 --seats A B
python -m nuru.avsd test --furhat 192.168.1.110 --seats A B   # live window showing who is speaking

# Fully automatic session: greeting, comments on each person, then C1 -> C2 -> C3 with auto-invitations
python -m nuru session --group G01 --conditions c3 c1 c2 --tasks T2 T3 T1 --minutes 3

# Or keep the operator panel, with automatic detection the operator only corrects
python -m nuru panel --avsd --auto
```

**How a session starts:** run the command, then walk away. The robot waits until someone has been in front
of it for about 2 seconds, then greets them by itself (it uses the camera detector, or Furhat's own face
tracking if the detector is down). Use `--start-now` to skip the wait while testing.

**If the camera or detector fails:** it restarts itself up to twice. If it is still down, the session keeps
going without it: the robot invites people on a timer instead of freezing, and the log records a
`detector_failed` event.

The `session` command pauses between conditions for the questionnaire (press Enter to go on, or use
`--no-pause`). Ctrl+C is the emergency stop. Useful options:

| Option | Meaning |
|---|---|
| `--camera furhat \| 0` | robot camera (default) or a webcam number |
| `--mic furhat \| default \| N` | robot mic (default), laptop mic, or a device number |
| `--threshold 9` / `--motion 0.004` | mic and mouth-motion sensitivity (lower = more sensitive) |
| `--questions 2` | discussion questions asked per condition |
| `--no-describe` | skip the expression and clothing comments after the greeting |

**Social features (`describe`):** after saying hello, the robot looks at each person and mentions the colour
of their top and their facial expression, adjusting the line (for example, a reassuring line for someone
who looks serious). These observations are written to the session log as `social_features` events.

The MediaPipe face model (`config/face_landmarker.task`) is downloaded automatically the first time.

---

## The three conditions in one picture

All three conditions use the same discussion script (`config/tasks.json`). The only thing that changes
is how the robot invites someone to speak:

```
                WHO is invited?            HOW is the invitation expressed?
C1 Non-adaptive  next seat in turn (A,B,C,D)  look at learner 3 s → "What do you think?" → wait 2 s
C2 Adaptive-conv quietest learner             look at learner 3 s → "What do you think?" → wait 2 s
C3 Adaptive-Nuru quietest learner             look at group → look at the pair → short look 1.5 s + soft nod
                                              → "Je, una maoni? What do you think?" → look back at group
                                              → wait 4 s → "Hakuna shida." (never insists)

      C1 vs C2 = effect of ADAPTIVITY          C2 vs C3 = effect of CULTURAL ADAPTATION
```

The code checks this design automatically: a unit test fails if C1 and C2 ever differ in anything except
targeting, or if C2 and C3 stop sharing the same participation model.

---

## Documentation

| File | What it explains |
|---|---|
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | **Start here.** Step-by-step guide for running a session, written for operators with no programming background |
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | Architecture and algorithms, mapped to each part of the abstract |
| [docs/STUDY_PROTOCOL.md](docs/STUDY_PROTOCOL.md) | Phase 1, 2 and 3 procedures, counterbalancing, measures, questionnaire, ethics |
| [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) | What we built and why, the power analysis, and recommendations |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Fixes for common problems |

## Repository layout

```
config/
  nuru.json          robot IP, voices, seat positions, moderation timing
  avsd.json          where each seat appears in the camera image (python -m nuru.avsd calibrate)
  profiles.json      the C1 / C2 / C3 behaviour profiles (edit C3 after Phase 1)
  tasks.json         greeting, 3 discussion tasks, closing
nuru/
  robot.py           fault-tolerant Furhat Remote API client (+ MockRobot)
  participation.py   participation model: speaking time, turns, Gini, invitee choice
  cues.py            gaze, nod and speech cue sequence for an invitation (+ custom gestures)
  controller.py      moderation engine: threads, auto-invite rules, logging
  vad.py             per-participant microphone voice-activity detection
  avsd.py            audio-visual speaker detection (Furhat camera + mic), expression/clothing
  panel.py + static/ Wizard-of-Oz operator web panel
  clips.py           Phase 1 co-design stimulus clips
  design.py          Williams-design counterbalancing (conditions × topics)
  analysis.py        logs → equity / engagement measures, statistical tests, figure
  simulate.py        synthetic groups: pipeline test and power analysis
  __main__.py        command line (python -m nuru …)
tests/test_nuru.py   14 unit and integration tests  (python -m pytest -q)
data/                questionnaire template
examples/            SYNTHETIC logs and results showing the analysis output (not real data)
logs/                session logs are written here (git-ignored)
nuru_prototype.py    the team's first prototype (kept for reference; superseded by nuru/)
```

## Status (18 Nov 2026)

* New (17 Nov): fully automatic sessions, camera-based speaker detection and social features
  (expression, clothing colour). The configuration is now set up for **2 seats (A, B)** and the robot at
  192.168.1.110. Auto-invite timing was also relaxed (15 s silence, at least 20 s between invitations).
* Verified live on the Furhat at 192.168.0.105: speech in English and Kiswahili, gaze, built-in
  and custom gestures, and full C1/C2/C3 invitation sequences driven from the operator panel.
* 14/14 automated tests pass.
* Open items for the team: calibrate seats with real participants, have native speakers check the
  Kiswahili prompts, replace the placeholder C3 values with the Phase 1 results, and decide the primary
  outcome (see [DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md#5-power-analysis--the-most-important-finding)).
