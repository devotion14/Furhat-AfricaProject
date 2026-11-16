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
| Run a study session | `python -m nuru panel`, then open http://localhost:5000 | Phase 3, Study A |
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

# 2. On the robot: start the "Remote API" skill (Furhat web UI -> Skills -> Remote API -> Start)
#    The robot's IP is in config/nuru.json (currently 192.168.0.105)

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
Checking Furhat at 192.168.0.105:54321
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
  profiles.json      the C1 / C2 / C3 behaviour profiles (edit C3 after Phase 1)
  tasks.json         greeting, 3 discussion tasks, closing
nuru/
  robot.py           fault-tolerant Furhat Remote API client (+ MockRobot)
  participation.py   participation model: speaking time, turns, Gini, invitee choice
  cues.py            gaze, nod and speech cue sequence for an invitation (+ custom gestures)
  controller.py      moderation engine: threads, auto-invite rules, logging
  vad.py             per-participant microphone voice-activity detection
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

## Status (16 Nov 2026)

* Verified live on the Furhat at 192.168.0.105: speech in English and Kiswahili, gaze, built-in
  and custom gestures, and full C1/C2/C3 invitation sequences driven from the operator panel.
* 14/14 automated tests pass.
* Open items for the team: calibrate seats with real participants, have native speakers check the
  Kiswahili prompts, replace the placeholder C3 values with the Phase 1 results, and decide the primary
  outcome (see [DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md#5-power-analysis--the-most-important-finding)).
