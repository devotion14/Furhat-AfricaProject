# DD2413 Lab 1 — grading guide for Nuru

This guide maps every line of the *DD2413 Lab 1 Grading Sheet 2026* to the code, the demo and the
answers we should be able to give. Each of us must be able to say all of it **in our own words**.
Part B is answered without help.

> **The one thing to remember:** for Lab 1 we demo `python -m nuru live`, **not** the operator
> panel. The panel is Wizard-of-Oz: a person marks who is speaking and starts the session, so it
> fails criterion 1 (E: "starts the interaction without the group triggering it by hand"). The live
> mode senses, decides and acts by itself.

---

## 0. Demo-day checklist

```powershell
python -m pytest -q                               # 21 tests should pass
python -m nuru check                              # robot reachable, voices, gestures, speech
python -m nuru live --mock --scenario group       # rehearsal without the robot (also: solo, newcomer, far, failures)
python -m nuru live                               # THE DEMO: then nobody touches the laptop
```

**Test the live mode on the real robot before the slot.** All five scenarios run in mock and in the
tests. They have not yet run on the robot. Two things only the real robot can show:

1. How long Furhat's `listen` waits in silence before it returns (its no-speech timeout). Time it and
   use that number in the Q6 answer.
2. Whether the face tracker gives stable ids at our distances. If people "flicker", raise
   `lost_after_s` in `config/live.json`.

**Demo script (about 6 minutes, gives evidence for every A descriptor):**

| # | Who | What happens | Shows |
|---|---|---|---|
| 1 | One team member walks up | Robot notices them, smiles, greets one-to-one, asks with direct gaze; on silence: *"Sorry, I didn't catch that"* + repeats | C1-E auto start, **solo value** of the feature, C4 repair |
| 2 | Two team members walk up together | Greeting sweep over both faces, bilingual group greeting, open question to the group, then invites the person not yet heard: group glance → pair glance → short gaze + soft nod → *"Je, una maoni? What do you think?"* → gaze back to group | **group value**: clearly different behaviour (C2-A) |
| 3 | During run 2, a third person joins | Robot turns to the newcomer, nods, *"Karibu! Welcome, please join us"*, and later invites them | who is present + how long they have been there |
| 4 | Invited person stays silent | *"Hakuna shida. That's okay, anyone can add."* The robot never insists | C4-A, cultural design |
| 5 | A person **not in our group** (TA or classmate) | Full run | "Runs reliably" (C1-A) |
| 6 | Everyone walks away | Robot ends the interaction, goes back to an idle look-around, and does not re-greet the same people | no freeze |

While it runs, keep the terminal visible. Every line is tagged `[SENSE]`, `[PLAN]` or `[ACT]`, so we
can point at the boundary between perception, decision and behaviour live (criterion 1-A).

Fallback if the hardware is down: the same demo on the Furhat SDK virtual robot (add virtual users in
the SDK web UI) counts for "runs at least once", plus the submitted video.

---

## 1. The pipeline (draw this on request)

```
 SENSE (perception)                       PLAN (decision)                     ACT (behaviour)
 ─────────────────────────────────        ──────────────────────────────      ─────────────────────────────
 Furhat camera                            nuru/live.py  plan_* functions      nuru/live.py act_*, nuru/cues.py
  └ on-robot face tracker (MODEL)         (hand-written rules, no robot        → Furhat Remote API (HTTP)
     in : camera image                     calls, unit tested)                   gaze  /attend
     out: [{id, x, y, z}] via /users                                             face  /gesture (Nod, NuruSoftNod,
 nuru/perception.py PeopleTracker         plan_engagement   idle/beckon/start          NuruWarmSmile)
     in : face list every 0.3 s           plan_mode         solo vs group       speech /say  en + sw voices
     out: Percept = who is present,       plan_between_q.   welcome/farewell      (Azure neural TTS = MODEL)
          distance, dwell time,           plan_invitee      least heard,        lights /led  state colour
          arrived / left                                    longest present
 Furhat mic → cloud ASR (MODEL)           plan_after_listen ack / repair /
     in : audio                                             release
     out: transcript → robot.listen_timed
          reduces it to speech | silence
          | error + duration (words dropped)
          ═══════════ perception ends here: Percept + listen status ═══════════
```

**Where each boundary is in the code:**

* Perception ends at the `Percept` dataclass (`nuru/perception.py`) and the dict returned by
  `robot.listen_timed` (`nuru/robot.py`). Nothing after this point sees raw sensor data.
* The decision is made in the `plan_*` functions at the top of `nuru/live.py`. They are pure: they
  take a Percept and return a `Decision(action, target, mode, reason)`.
* Behaviour starts in `LiveSession.act_*` and `ask()`. Every call there goes to the robot.

**Hand-written vs model:**

| Component | Hand-written or model |
|---|---|
| Face detection and tracking | model, pretrained, runs on the robot |
| Speech recognition | model, cloud, used only as a speech/silence detector |
| Speech synthesis (AsiliaNeural en-KE, ZuriNeural sw-KE) | model, Azure neural TTS |
| Perception filtering, all decision rules, cue sequences, timing | hand-written |

There is no language model. The robot speaks only fixed lines from `config/live.json`,
`config/tasks.json` and `config/profiles.json`.

---

## 2. Part A, criterion by criterion

### 1. Sense–plan–act loop (target: A)

| Level | Evidence | Where |
|---|---|---|
| E: detects a person and starts without a hand trigger | A face confirmed for 1 s within 1.6 m → `start` | `plan_engagement`, `LiveSession.run` |
| C: explain every component incl. model calls | §1 table: inputs and outputs of the face tracker, ASR, TTS | §1 |
| A: point to where perception ends / decision / behaviour | `[SENSE]` / `[PLAN]` / `[ACT]` terminal tags; three code sections | `perception.py` → `live.py` PLAN → `live.py` ACT |

### 2. Social adaptation (target: A)

* **The feature:** the number of people present, from the camera, plus who arrived when and who
  has been heard. It is never based on what people say. The ASR transcript is discarded in
  `listen_timed`; we keep only spoke / silent / error and the duration.
* **Visible effect:**

  | | 1 person (solo) | 2+ people (group, Nuru C3 cues) |
  |---|---|---|
  | Greeting | direct gaze, smile, *"Habari! Hello…"* | gaze sweep over each face, soft nod, *"Habari zenu! Karibuni…"* |
  | Question | looks at the person | looks at the group centre (open floor) |
  | Turn allocation | the person | invites the least-heard person (ties: present longest), never the same person twice in a row |
  | Invitation cues | — | group glance 1 s → pair glance 0.8 s → individual gaze 1.5 s + soft nod → sw + en prompt → gaze back to group |
  | Silence | *"Sorry, I didn't catch that. Take your time"* + repeat, then move on | *"Hakuna shida. That's okay, anyone can add."* Never repeats to an individual |
  | Newcomer | switches to group mode | *"Karibu! Welcome, please join us"* |

* **Two contrasting runs:** demo rows 1 and 2.
* **Why it matters for the project:** Nuru is for high-ratio East African classrooms, where being
  singled out in front of peers is face-threatening. Phase 1 co-design hypothesises group-first,
  indirect, bilingual cues. One-to-one, a direct style is natural. In a group, the robot must spread
  the floor and protect quiet learners. The same adaptation (perceived group, then culturally
  adapted invitation) is what the C2 vs C3 comparison in the study tests.

### 3. Use of embodiment (target: A)

* **Modalities, coordinated in time:** gaze/head (`/attend`), facial gestures (`NuruSoftNod`,
  `NuruWarmSmile`, `Nod`), speech in two languages and two voices, LED ring colour. In an invitation
  the gaze shift comes first, then the nod (blocking, so it finishes before speaking), then the
  prompt, then the gaze returns to the group while the robot listens (`act_invite`).
* **Why these, for this robot:** Furhat has an expressive projected face and a 3-DOF neck, but no
  arms and no locomotion. Turn allocation therefore has to be done with gaze and head orientation,
  which is also the channel the co-design work is about. Nods and closed smiles give backchannel
  without words. Kiswahili and English with Kenyan voices signal cultural belonging. The LED shows
  the robot's state (blue = listening) because the face is busy with social signals.
* **Explicit mapping:** decision → behaviour is a fixed table in code, the `act_*` functions plus
  the profile values in `config/profiles.json` (`c3`) and `config/live.json` (`solo`, `led`,
  `lines`). No model chooses a behaviour.

### 4. Robustness and timing (target: A)

| Failure | What happens (by design) | Where |
|---|---|---|
| Silence | Solo: repair + repeat once, then *"That's okay, let's try another one"*. Group: gentle release, next question | `plan_after_listen` |
| ASR / network error | *"Sorry, I could not hear properly"*, retry once (solo), then move on | `listen_timed` returns `error`; `plan_after_listen` |
| Slow / hung model call | Every HTTP call has a timeout; `listen` is hard-capped at 20 s, then `/listen/stop` | `robot._call`, `listen_timed` |
| Robot call fails | Returns False, never raises; the loop catches any exception, logs it and continues | `robot._call`, `LiveSession.run` |
| Camera hiccup | Last known people are kept; 1 s confirmation and 3 s loss hysteresis | `PeopleTracker.update` |
| Person leaves mid-turn | Detected after listening; the interaction ends politely, robot returns to idle | `ask()`, `plan_between_questions` |
| Passers-by / far people | Ignored beyond 2.5 m; between 1.6 and 2.5 m the robot beckons once | `plan_engagement` |
| Waiting without feedback | LED blue while listening; immediate nod + *"Asante"* after speech; gaze on the addressee | `listen`, `act_acknowledge` |

No step needs a restart by the group. `tests/test_nuru.py` runs full sessions with no operator
(`test_live_session_runs_without_operator`).

---

## 3. Part B — oral questions (practise saying these, don't read them)

**Q1. What makes this interaction social?**
The robot reacts to the social situation, not to the words. It notices how many people are there,
who has just joined and who has not been heard yet. It then manages turn-taking with gaze, nods and
politeness strategies: group first, never singling someone out, never insisting. Its goal is
equitable participation, which is a social outcome, not an information task.

**Q2. The sense–plan–act loop, non-technically; hand-written vs model.**
The robot's eyes see faces, and its ears hear whether someone spoke. Our rules decide: one person
means chat directly; a group means open the question to everyone and gently invite whoever hasn't
spoken. Its face, head, voice and lights then carry that out. The models are the face tracker,
speech recognition and the voices. Everything that decides is hand-written (see the table in §1).

**Q3. What does the model receive, and how is its output constrained before the robot acts?**
The face tracker receives camera images, on the robot. We receive only ids and 3-D positions, and we
filter them: malformed entries are dropped, nothing beyond 2.5 m counts, a face must be seen for 1 s
to count and missing for 3 s to be gone. The ASR receives audio. Its transcript is reduced to
speech / silence / error, so no recognised word can change what the robot does. The TTS receives
only our fixed scripted lines. No model output is spoken or turned into an action directly.

**Q4. What data leaves the laptop or robot, where does it go, and did the people in the video know?**
Camera images stay on the robot. Face positions go robot → laptop over the local network. Audio goes
robot → the cloud speech-recognition service configured on the Furhat, and the transcript comes
back. Our code discards it. The robot's lines go to Microsoft's Azure TTS as text. The laptop CSV log
(`logs/nuru_LIVE_*.csv`) holds anonymous face ids, times, durations and the robot's lines: no
images, audio or transcripts.
*Each of us must answer the last part truthfully* (who was filmed, whether they were told and
agreed). Check which recogniser the robot uses (web UI → settings) before the demo.

**Q5. One ethical risk and what to do in the final project.**
The risk is that a robot that tracks who speaks least becomes a surveillance and grading tool, or
publicly marks a learner as "the quiet one". That causes stigma and face loss, which is worst in
exactly the hierarchical classrooms we target. It is also a risk to assume all East African
learners share one interaction style (cultural stereotyping). What we would do in the final
project:

* No per-student participation reports to teachers.
* Opt-out and assent from children, consent from parents and the school.
* Keep the no-repeat and never-insist rules.
* Validate the cue set with the co-design sessions instead of our assumptions.
* Replace cloud ASR with on-device voice activity detection (`nuru/vad.py`), so children's audio
  never leaves the room.

**Q6. Least reliable part, and what if the user says nothing for ten seconds?**
The least reliable part is speech detection through the cloud ASR, which suffers from network
latency, classroom noise, Kenyan English, and code-switching with an `en-US` model. In a group the
robot also can't tell *who* answered an open question, because the Remote API gives no sound
direction. We only attribute speech to a person right after inviting them.
Ten seconds of silence: the robot keeps looking at the person with a blue LED (it is visibly
listening). Furhat's `listen` returns "silence" after its timeout (measure it on the robot), and
then:

* Solo: *"Sorry, I didn't catch that. Take your time"*, repeat the question, listen again, then
  *"That's okay, let's try another one"*.
* Group: *"Hakuna shida. That's okay, anyone can add"*, then the next question.

It never freezes, and listening is hard-capped at 20 s.

**Q7. Something realistic the robot's limitations made impractical.**
Knowing *who* is speaking in a group from the robot alone. The Furhat Remote API gives no
microphone direction and no per-face "is talking" signal. Real equity tracking (the C2/C3
participation model in `participation.py`) needs a lapel mic per learner (`vad.py`) or the
wizard panel. Others:

* Furhat has no arms, so it can't point or beckon with a hand.
* It can't move closer to a learner.
* Its face tracker handles a few people, not a 40-pupil classroom.
* Robust Kiswahili or code-switched recognition isn't available, so we only detect *that*
  someone spoke.

**Q8. What did you personally build? Show it in the code or prompts.**
Each of us prepares this alone. Pick your files and functions, open them, and be able to walk
through one in detail: what goes in, what comes out, and one design choice you made and why. Don't
claim a part you can't explain line by line. If an AI tool helped write it, say so, and show that you
understand and control what it does.

| Student | Built | Show in code |
|---|---|---|
| Luke Tsalwa | | |
| Celestino M.K. Kariuki | | |
| Devotha R. Lyakurwa | | |

Possible split, so each person has a clear area. Adjust it to what you actually did:

* Perception: `perception.py`, `robot.listen_timed`
* Decision rules and tests: `live.py` `plan_*`, `tests/`
* Behaviour and cues: `cues.py`, `profiles.json` C3, `live.py` `act_*`, voices

---

## 4. Tuning knobs (config/live.json)

| Key | Default | Effect |
|---|---|---|
| `engage_distance_m` | 1.6 | Closer than this → start; further (up to `max_distance_m`) → beckon |
| `confirm_s` / `lost_after_s` | 1.0 / 3.0 | Presence hysteresis; raise `lost_after_s` if faces flicker |
| `listen_timeout_s` | 20 | Hard cap on one listen |
| `n_questions` | 2 | Questions per interaction (keep the demo short) |
| `task` | T3 | Which discussion task from `tasks.json` |
| `group_profile` | c3 | Cue profile used in group mode (try `c2` to show the non-cultural version) |
| `lines` | — | Everything the robot says outside the task questions |
