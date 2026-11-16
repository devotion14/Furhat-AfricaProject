# How Nuru works

This document explains the system from the top down and maps each part to the research abstract.

## 1. The big picture

```
   INPUT: who is speaking?                BRAIN                                 OUTPUT: the robot
 ┌─────────────────────────┐   speak(A)  ┌────────────────────────────────┐   HTTP  ┌──────────────────┐
 │ Operator panel (WoZ)    │────────────►│ NuruController                 │────────►│ Furhat Remote API│
 │  keys 1-4 / Space       │  silence()  │  ├ Tracker (participation model)│  :54321 │  attend (gaze)   │
 ├─────────────────────────┤────────────►│  ├ behaviour profile C1/C2/C3   │         │  say (en / sw)   │
 │ Microphones + VAD       │             │  ├ auto-invite rules            │         │  gesture (nod)   │
 │  (optional, automatic)  │             │  └ EventLog (CSV)               │         └──────────────────┘
 └─────────────────────────┘             └───────────────┬────────────────┘
                                                         │ logs/nuru_G01_*.csv
                                                         ▼
                                         analysis.py → results/report.md, figure
```

The input side does not affect the logic. Speech events from the panel and from the microphones go
through the same two calls, `speak(seat)` and `silence()`. This is how the abstract's "Wizard-of-Oz
fallback if sensing proves unreliable" is implemented: either source can feed the system, and the
operator can always correct it.

## 2. From abstract to code

| Abstract says | Implemented in | How |
|---|---|---|
| "A participation model tracks each learner's speaking time and turns" | `participation.py` → `Tracker` | Accumulates seconds and turns per seat, including the turn in progress |
| "per-participant microphones with voice-activity detection" | `vad.py` | Energy VAD with adaptive noise floor, crosstalk suppression and hysteresis |
| "Furhat's face tracking for location" | `controller.seats_from_users`, `python -m nuru calibrate --faces` | Detected faces are sorted left to right and assigned to seats |
| "Wizard-of-Oz fallback" | `panel.py`, `static/panel.html` | Web panel with keyboard shortcuts |
| C1 "gaze follows the current speaker, evenly cycled attention between turns" | `profiles.json: c1` (`policy: round_robin`, `idle_gaze: cycle`) | Follows the speaker; between turns the gaze cycles A→B→C→D |
| C2 "directs gaze and head towards the least active learner, conventional cues" | `profiles.json: c2` (`least_active`) | Same cue form as C1, different target |
| C3 "same participation model as C2, expressed through the Phase 1 behaviour set" | `profiles.json: c3`, `cues.py` | Same policy as C2, different cue sequence |
| Phase 1 clips varying gaze duration, target, head orientation, wait time, invitation style | `clips.py` (V00–V11) | One factor changed at a time from a baseline |
| "shorter individual gaze… broader gaze to the group" | `gaze_hold_s: 1.5`, `gaze_during_wait: group` | |
| "inviting the whole group or a neighbouring pair before an individual" | `group_glance_s`, `pair_glance_s` | The pair is the invitee plus their nearest seat |
| "longer wait time after a question" | `wait_s: 4.0` (vs 2.0) | |
| "acknowledging nods" | custom gesture `NuruSoftNod` | Small, slow nod with a closed smile, also used after a response |
| "brief bilingual (English/Kiswahili) prompts" | `prompt_mode: bilingual`, voices `ZuriNeural` (sw-KE) and `AsiliaNeural` (en-KE) | Kenyan voices; the translation stays aligned with the English |
| "the robot never insists on an answer" | `release` lines and the no-repeat window | One invitation per person, a gentle release, then move on |
| "Latin-square order and topics balanced" | `design.py` | Williams design (6 orders), topic fixed by position |
| Gini of time and turns, least-active share, response rate and latency | `participation.metrics`, `analysis.py` | |
| "repeated-measures tests or mixed models" | `analysis.inferential` | Friedman test plus planned Wilcoxon contrasts; the CSV is ready for mixed models in R |

## 3. The participation model (`participation.py`)

* **Speaking time and turns.** `start(seat)` opens a turn and `stop()` closes it. A new speaker
  implicitly closes the previous turn.
* **Gini coefficient.** `G = 2·Σ i·x_(i) / (n·Σx) − (n+1)/n` over the sorted values. With four people,
  0 means perfectly equal and 0.75 means one person did all the talking.
* **Choosing whom to invite.**
  * `round_robin` (C1): A, B, C, D, A, …, regardless of who has spoken.
  * `least_active` (C2, C3): the person with the least speaking time. Ties are broken by fewest
    turns, then fewest invitations. Two rules prevent singling someone out: the current speaker is
    never chosen, and anyone invited in the last `no_repeat_window` (2) invitations is skipped. A
    quiet person cannot be invited twice in a row.
* **Responses.** When the robot finishes its prompt, a response window opens
  (`response_window_s` = 8 s, the same in every condition so the measure is fair). If the invitee
  starts speaking inside the window, that counts as a response, with latency measured from the end
  of the prompt. Answering before the prompt ends counts as latency 0. All other turns are counted
  as voluntary.

## 4. The invitation cue sequence (`cues.py`)

One function, `perform_invitation`, produces every condition and every Phase 1 clip. Only the
profile data changes:

```
1 group glance (s)  →  2 pair glance (s)  →  3 orient head/eyes (fraction)  →  4 smile / nod
→  5 prompt (sw then en)  →  [response window opens]  →  6 hold gaze (s)
→  7 wait, looking at the group or the individual (s)  →  8 release if nobody answered
```

Steps 6–8 stop as soon as anyone starts speaking. The robot never talks over a participant. If the
invitee answered, the robot looks at them and, in C3, gives an acknowledging nod.

Measured on the real robot (16 Nov), a full C3 invitation takes about 8 s to the end of the prompt:
1.0 s group glance, 0.8 s pair glance, 1.3 s nod, about 2.7 s Kiswahili and 1.2 s English. A C2
invitation takes about 1.9 s.

## 5. The controller (`controller.py`)

Three threads keep the operator panel responsive:

* **Caller threads** (panel, keyboard, VAD) update the tracker instantly.
* **One robot worker** runs robot actions in order from a queue, so gaze commands never conflict. A
  newer gaze target makes older queued gaze commands obsolete.
* **A supervisor** checks five times a second. It closes unanswered invitations (logged as
  `no_response`) and runs the auto-invite rules:
  * **lull**: the group has been silent for 7 s;
  * **turn_end**: a turn just ended (1.5 s of silence) and at least 40 s have passed since the last
    invitation.

  At least 12 s must always pass between invitations. See DESIGN_DECISIONS §5 for why the
  turn-end rule matters.

Robot network errors are logged and never crash a session. **STOP** (Esc) clears the queue, stops
speech and turns the robot back to the group.

## 6. Voice-activity detection (`vad.py`)

For each 30 ms frame and each microphone channel:
1. Compute energy in dB.
2. Track the noise floor: it drops immediately to quieter levels and rises slowly (0.5 dB/s).
3. Mark the channel active if it is more than 12 dB above its floor.
4. Suppress crosstalk: only the channel with the highest margin counts, plus any channel within
   6 dB of it (genuine overlapping speech).
5. Apply hysteresis: a turn starts after 250 ms of speech and ends after 800 ms of silence.

Where speech overlaps, the floor belongs to whoever started first. A unit test checks that a loud
speaker with a quieter neighbour's crosstalk produces exactly one turn for the right person.

## 7. Logging (`eventlog.py`)

Every event is one CSV row, flushed to disk immediately:
`t, t_rel, group, condition, task, event, seat, value`.
Events include `speak_start`, `speak_stop`, `invite`, `cue`, `robot_say` (logged at speech onset, to
line up with video), `prompt_end`, `response`, `no_response`, `task_question`, `metrics`, and
`participant_meta`.

`analysis.py` **recomputes every measure from the raw events** rather than trusting the live
counters, so a crashed session can still be analysed. A test checks that the live and recomputed
values agree.

## 8. Talking to Furhat (`robot.py`)

Furhat's Remote API is a REST server on port 54321. We use:

| Call | Used for |
|---|---|
| `POST /furhat/attend?location=x,y,z` | gaze and head (metres: x left/right, y up/down, z forward) |
| `POST /furhat/say?text=…&blocking=true` | speech; returns when finished |
| `POST /furhat/voice?name=…` | switching between `AsiliaNeural` (en-KE) and `ZuriNeural` (sw-KE); about 0.1 s |
| `POST /furhat/gesture?name=Nod` or a JSON body | built-in or custom gestures |
| `POST /furhat/say/stop` | emergency stop |
| `GET /furhat/users` | face tracking, for calibration |

Custom gestures are sent as JSON keyframes with `"class": "furhatos.gestures.Gesture"`. Using
`furhat.gesture.Gesture` is rejected by the robot (we found this out during testing).

## 9. Porting to other robots

All robot-specific code is in `robot.py`, which has nine methods. To run Nuru on NAO, Pepper or a
tablet avatar, write a class with the same methods (`attend_location`, `say`, `gesture`, …). Nothing
else needs to change, which supports the abstract's portability claim.
