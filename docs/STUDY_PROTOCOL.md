# Study protocol (operational)

This document turns the abstract's Method section into concrete procedures that use this software.

## Phase 1 — Participatory behaviour design

**Goal:** a small, validated cue set for C3.
**Sessions:** 2–3 sessions of 4–6 East African students and teachers each, in person or remote.

1. **Prepare the stimuli.** Film the 12 clips (`python -m nuru clips all`) from the invited
   learner's viewpoint, and also from a side view. Each clip changes **one** cue relative to the
   conventional baseline V00:

   | Dimension | Clips |
   |---|---|
   | gaze duration | V01 short 1.5 s, V02 long 5 s |
   | gaze target | V03 group first, V04 neighbouring pair first, V05 whole group only |
   | head orientation | V06 partial head turn |
   | wait time | V07 5 s, with gaze returned to the group |
   | invitation style | V08 bilingual, V09 soft nod and smile, V10 explicit permission to pass |
   | bundle | V11 current C3 candidate |
2. **Run the session.** Show V00 first as the reference, then the other clips in a different random
   order for each session. For each clip, ask participants to rate on a scale of 1–7:
   * *How comfortable would you feel if the robot did this to you?*
   * *How appropriate is this in a classroom you know?*
   * *Would this make you want to speak?*
   * *Would this make you feel singled out?*

   Then ask them to rank the clips and discuss.
3. **Analyse.** Keep a cue if it improves comfort and appropriateness over V00 without reducing
   willingness to speak. Treat each candidate adaptation as a hypothesis, not as a fact about any
   culture.
4. **Update C3.** Edit `config/profiles.json → c3` with the chosen values. Have native speakers
   check the Kiswahili wording. Do not change C1 or C2.

## Phase 3 — Study A (live, small groups)

* **Participants:** groups of 4 with East African schooling backgrounds. Target 8–10 groups (see
  DESIGN_DECISIONS §5 on power).
* **Design:** within-subjects, three conditions. Run `python -m nuru plan --groups 10` for the
  Williams design (balanced for both position and carry-over), with topics paired by position.
  Each condition × topic pairing occurs equally often in every set of 6 groups.

| Time | Step | Panel action |
|---|---|---|
| −10 min | Consent; pre-session self-report of usual participation | Enter `quiet / average / active` per seat |
| 0 | Greeting | **Start session**, then **Greeting** |
| 1–11 | Block 1: task intro and 4 questions | **Start C…**, then **N** ×4, marking speech throughout |
| 11–14 | Post-condition questionnaire | **End condition** |
| 14–27 | Block 2 | same as Block 1 |
| 27–40 | Block 3 | same as Block 1 |
| 40–50 | Final questionnaire and short semi-structured interview | **Closing**, then **End session** |

**Keep these constant across conditions:** invitation mode (auto-invite recommended), block length
(about 10 minutes), operator, room, and seating.

## Phase 3 — Study B (scaled, video-based)

Record classroom-scale scenarios for each configuration. Put any number of seats in
`config/nuru.json` (for example a 6 × 9 grid). The participation model and cue code work for any
number of seats, and the "pair" becomes the nearest neighbour. Teachers and students in Kenya and
Tanzania then rate the videos online, using the same appropriateness and comfort items as Phase 1.

## Measures

| Construct | Measure | Source |
|---|---|---|
| Participation equity (RQ1) | Gini of speaking time; Gini of turns; least-active member's share | `analyze` |
| Effects on quieter learners (RQ3) | share of time and turns for self-reported quiet learners; their response rate | `analyze` (uses `participant_meta`) |
| Engagement | voluntary turns; response rate; response latency | `analyze` |
| Perceived inclusion and pressure (RQ3) | Likert items below | questionnaire |
| Cultural acceptability (RQ2) | appropriateness and comfort items; interviews analysed thematically | questionnaire, interviews |

### Post-condition questionnaire (Likert 1–7, from strongly disagree to strongly agree)

| item code | Statement |
|---|---|
| `fair_attention` | The robot shared its attention fairly among us. |
| `felt_invited` | I felt invited to share my ideas. |
| `felt_pressured` | I felt pressured to speak. *(RQ3)* |
| `singled_out` | I felt singled out by the robot. *(RQ3)* |
| `appropriate` | The robot's behaviour was appropriate for a classroom I know. *(RQ2)* |
| `comfortable` | I felt comfortable with how the robot looked at me. *(RQ2)* |
| `respectful` | The robot was respectful. *(RQ2)* |

Enter the answers in `data/questionnaire.csv` (copy `data/questionnaire_template.csv`), one row per
person, condition and item. `python -m nuru analyze logs/ --questionnaire data/questionnaire.csv` then
adds them to the report.

### Statistics
* Primary: Friedman test across C1, C2 and C3, plus two **planned** Wilcoxon contrasts: C1 vs C2
  (adaptivity) and C2 vs C3 (cultural adaptation). No other pairwise tests are run, so there is no
  fishing.
* Recommended for the final paper: a mixed model with group as a random effect, using
  `summary_by_block.csv`, for example in R:
  `lmer(gini_time ~ condition + position + (1|group))`; for Likert items, use `clmm` at the
  individual level.

## Ethics, built into the software

| Principle (abstract) | Implementation |
|---|---|
| The robot never insists | At most one invitation at a time; the same person is never invited twice in a row; a gentle release line; the robot never repeats the question |
| Silence is not disengagement | No-response events are only logged, and the robot moves on without comment |
| Teacher or operator stays in control | STOP button (Esc); manual or auto mode; free-text Say |
| Voluntary participation | The greeting states that participants can always pass |
| Data minimisation | Logs contain seat letters only, never names; no audio is recorded |
