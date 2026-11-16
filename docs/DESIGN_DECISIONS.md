# What we did, and why

## 1. Starting point
The team's `nuru_prototype.py` was a sound proof of concept: a keyboard tracker, the three
conditions, and a Gini calculation. To run a real study, we turned it into the `nuru` package and
made several methodological improvements along the way.

## 2. Improvements over the prototype (and why each matters)

| # | Prototype | Now | Why it matters for the research |
|---|---|---|---|
| 1 | C1 held gaze 2 s, C2 held 3 s | C1 and C2 have **identical** cue form | Otherwise the C1 vs C2 comparison would mix adaptivity with gaze length, and RQ1's first part could not be interpreted. A unit test now enforces this. |
| 2 | Robot blocked the keyboard during an invitation | Separate robot worker thread | The operator can mark speech during the 5–8 s invitation, so responses and their latency are measured. |
| 3 | Invitation always ran to the end | Stops as soon as someone speaks | The robot must not talk over a learner (ethics), and latency is measured accurately. |
| 4 | No response measure | Response window, latency, voluntary vs invited turns | These are engagement measures named in the abstract. |
| 5 | No idle-gaze difference | C1 cycles, C2 looks at the least active, C3 looks at the group | Implements "evenly cycled attention between turns" (C1) and participation-aware gaze (C2). |
| 6 | Kiswahili spoken with the English voice | sw-KE voice (`ZuriNeural`) and en-KE voice (`AsiliaNeural`) | Correct pronunciation; regional English accent. |
| 7 | Built-in Nod only | Custom `NuruSoftNod`: smaller, slower, with a smile | A more acknowledging, less demanding nod, which is the candidate adaptation. |
| 8 | Ordering not specified | Williams design (6 orders) with topics balanced | Controls order and carry-over effects. |
| 9 | Gini on the console only | Full event log, offline analysis, tests, figure | Reproducible and auditable results. |
| 10 | Seat positions guessed | Face-tracking calibration plus a gaze-check sweep | Gaze cues only work if the robot looks at the right person. |
| 11 | Keyboard only | Wizard-of-Oz web panel, with optional microphone VAD | Matches the abstract's sensing plan, with the WoZ fallback. |
| 12 | Phase 1 not supported | 12 clips, each changing one factor at a time | Clip rankings can be attributed to a single cue. |

## 3. Choices we made (and the alternatives)
* **Remote API over a Kotlin skill.** The robot is driven from Python over REST. This keeps the whole
  study (sensing, logic, logging, analysis) in one language the team already uses, and works with
  both the real and virtual Furhat. The cost is about 50 ms latency per call, which is negligible
  for gaze shifts.
* **Same response window (8 s) in every condition.** The robot's wait time differs (2 s vs 4 s),
  but the measurement window does not. Otherwise C3 would get credit simply for waiting longer.
* **No-repeat window of 2.** This means the quietest person is never invited twice in a row. It
  slightly weakens the adaptive policy, but it implements "never single out / never insist".
* **The pair glance uses the nearest seat.** This works for four seats and for a 50-seat grid
  (Study B).

## 4. Verified on the real robot (16 Nov 2026, Furhat at 192.168.0.105)
* `python -m nuru check`: all 8 checks passed (Remote API, both voices, Nod, custom gesture, gaze,
  English and Kiswahili speech).
* `python -m nuru demo`: C1 invited A (round robin); C2 and C3 invited C, the least active seat.
* A C3 session driven through the operator panel's API: the timeline was group glance → pair glance
  (C+D) → orient → soft nod → "Una maoni gani?" → "What is your view?" → response at 0.49 s →
  acknowledging nod. Metrics and logs were correct.
* 14/14 automated tests pass (`python -m pytest -q`).

## 5. Power analysis — the most important finding
`python -m nuru simulate --power` runs the real policy code on synthetic groups of four (one
dominant, two average and one quiet member), many times over. The figures below are the share of
simulated studies in which each planned contrast is significant (α = .05).

**(a) If the robot only invites after a 7 s silence**, it gets the floor only about 8 times in
10 minutes, because lively groups rarely fall silent. Power for any Gini contrast was **≤ 0.09**
with 8–10 groups. The robot hardly acts, so it cannot change equity.
→ **Fix (implemented):** add a *turn-end* rule. The robot may also invite at a natural turn
boundary once 40 s have passed since its last invitation, giving about 15 invitations per 10 minutes.

**(b) With the turn-end rule**, assuming C3 cues raise the chance that an invited learner answers
from 0.5 to 0.8:

| Measure | 10 groups: C1 vs C2 | 10 groups: C2 vs C3 | 16 groups: C2 vs C3 |
|---|---|---|---|
| Gini speaking time | 0.13 | 0.15 | 0.19 |
| Gini turns | 0.21 | 0.21 | 0.36 |
| **Response rate to invitations** | 0.11 | **0.83** | **1.00** |

With no true C2/C3 difference, the false-positive rate was about 0.07, close to α, so the method is
sound.

**What this means for the study:**
1. A cultural-adaptation effect shows up first and most clearly in **whether learners answer the
   robot's cue** (response rate and latency), not in group-level Gini. This is exactly the question
   Asadi & Fischer (2025) raised: are non-verbal cues weak in themselves, or just badly tuned?
   **Recommendation: pre-register response rate to invitations (overall, and for quiet learners)
   as the primary outcome for C2 vs C3, with Gini as the primary outcome for the overall RQ1
   question and secondary for C2 vs C3.**
2. With 8–10 groups, a Gini difference is unlikely to reach significance unless the effects are
   large. Report effect sizes and confidence intervals, and frame Gini results as estimates.
   Longer blocks or more invitations increase power.
3. C1 vs C2 is small in groups of four, because round robin already reaches the quiet member 25 %
   of the time. That is an honest property of 4-person groups. The difference grows with group size,
   which is one argument for Study B.

*Caveat:* the simulated humans are crude, and these numbers depend on the assumed behaviour. Re-run
the simulation with pilot data (`--p-respond`) once the first groups are done.

## 6. Open items for the team
- [ ] Calibrate the seats with real people in the study room (`calibrate --faces`).
- [ ] Native-speaker check of all Kiswahili lines (`config/profiles.json`, `config/tasks.json`).
- [ ] Phase 1: replace the placeholder C3 values with the co-designed set.
- [ ] Pilot 1–2 groups and check operator marking reliability. Ideally a second person codes the
      video for one session, and Cohen's κ is computed on turns.
- [ ] Decide the primary outcomes before data collection (see §5).
- [ ] Ethics approval; consent forms in English and Kiswahili.
