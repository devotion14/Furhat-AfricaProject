# Nuru User Manual

This guide is written for whoever runs a session (the "operator" or "wizard"). You do not need to know
how to program. Follow the steps in order.

---

## Part 0 — Words used in this manual

| Word | Meaning |
|---|---|
| **Furhat** | The robot head with a projected face. It can speak, move its head and eyes, and show expressions. |
| **Remote API** | A small program ("skill") on the robot that lets our laptop control it over Wi-Fi. |
| **Operator panel** | The web page you use during a session to tell Nuru who is speaking. |
| **Seat A, B, C, D** | The four participant chairs, from left to right **as seen from behind the robot**. |
| **Condition C1/C2/C3** | The three robot behaviour styles we compare (see the README). |
| **Task T1/T2/T3** | The three discussion topics (water project, class trip, library). |
| **Invitation** | When the robot turns to someone and asks for their opinion. |
| **Gini** | A number from 0 to 0.75 for four people. 0 means everyone spoke equally; higher means less equal. |
| **Wizard-of-Oz** | You, the operator, mark who is speaking. The robot decides everything else. |

---

## Part 1 — One-time setup (about 20 minutes)

### 1.1 Install the software on the laptop
Open **PowerShell** in the project folder and run:
```powershell
pip install -r requirements.txt
```

### 1.2 Connect to the robot
1. The laptop and the robot must be on the **same Wi-Fi network**.
2. Open the robot's web interface in a browser: `http://192.168.0.105:8080`.
3. Go to **Skills**, find **Remote API**, and press **Start**.
4. If the robot has a different IP address, open `config/nuru.json` and change `"host"`.

### 1.3 Check everything works
```powershell
python -m nuru check
```
Every line should show `[OK ]`. You will hear the robot say "Nuru check" and "Habari", and see it nod.
If anything shows `[FAIL]`, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### 1.4 Set up the room
```
                 [ FURHAT ]
                     |
          about 1.2 m in front of the robot
                     |
      A          B         C          D        <- chairs in a gentle arc
     ←0.6 m→ ←0.4 m→  ←0.4 m→  ←0.6 m→           (left to right, seen from behind the robot)

   Operator laptop: behind or beside the robot, where you can see every face.
```
* Seat the robot at roughly the participants' eye height.
* Put a camera behind the participants if you are filming, so it captures the robot's face.

### 1.5 Aim the robot's gaze at the seats (calibration)
**Option A — automatic.** Ask four people to sit in the chairs and face the robot, then run:
```powershell
python -m nuru calibrate --faces
```
The robot finds the four faces and saves their positions in `config/nuru.json`.

**Option B — manual.** Run:
```powershell
python -m nuru calibrate
```
The robot looks at each seat in turn and says its letter. If it looks too far left or right for a seat,
open `config/nuru.json` and change that seat's **first number** (x) by about 0.1, then run the
command again. (If left and right appear swapped, swap the signs of the x values.)

---

## Part 2 — Running a session (about 45 minutes per group)

### 2.1 Before participants arrive
1. Start the panel:
   ```powershell
   python -m nuru panel
   ```
2. Open **http://localhost:5000** in Chrome or Edge.
3. In the top bar, check that the robot pill is **green ✓**.

### 2.2 Session setup (box "1 · Session")
1. Type the **group number** (`G01`, `G02`, …) and press **Load plan**. The panel fills in the correct
   condition order and topics for this group, which keeps the study counterbalanced.
2. For each seat, choose the participant's **usual participation** from the pre-session self-report:
   `quiet`, `average` or `active`. This is needed for research question 3.
3. Press **Start session**. A new log file is created, and its name appears under the buttons.
4. Press **Greeting**. The robot welcomes the group in Kiswahili and English and explains that
   participants can always pass.

### 2.3 Run each condition block (box "2 · Condition blocks")
Repeat for Block 1, Block 2 and Block 3:

1. Press **Start C…** for the block. The robot introduces the task.
2. Press **Next question (N)**. The robot asks the first question to the whole group.
3. **While people talk, mark who is speaking:**
   * press **1, 2, 3, 4**, or click the seat tile, when that person **starts** speaking;
   * press **Space** when **nobody** is speaking.

   The seat tile turns **green** for the current speaker. The robot looks at whoever is speaking.
4. **Invitations.** Either:
   * press **I** (Invite next) when there is a natural pause, and the robot picks who to invite
     according to the condition; or
   * tick **auto-invite** in the top bar, and the robot invites by itself after a pause.

   **Use the same method in all three blocks for a group.** We recommend auto-invite.
   While the robot is inviting, the seat tile turns **amber**. When the invited person starts
   talking, press their number as usual. Nuru records it as a response and measures the reaction time.
5. After about 2–3 minutes, press **N** for the next question. There are four questions per task.
6. When the block ends (about 10 minutes), press **End condition**. The block's results appear in
   the table under "Live participation measures".
7. Hand out the **post-condition questionnaire** (see [STUDY_PROTOCOL.md](STUDY_PROTOCOL.md)).
   **Do this before starting the next block.**

### 2.4 End the session
1. Press **Closing**. The robot thanks the group.
2. Press **End session**. The log is saved in the `logs/` folder.

### 2.5 Keyboard shortcuts

| Key | Action |
|---|---|
| `1`–`4` | Seat A–D started speaking |
| `Space` | Silence |
| `I` | Invite (the robot chooses who) |
| `N` | Next question |
| `Esc` | **STOP** the robot immediately |

Other useful controls:
* **invite** on a seat tile invites that specific person. This bypasses the policy, so use it only if
  the protocol requires it.
* **look** on a seat tile makes the robot look at that person.
* **Say** makes the robot speak any text you type, for example "Let's take a short break."
* **Log note** records an observation without the robot saying anything, for example
  "B left the room".

### 2.6 Rules of thumb for good marking
* Mark a **new turn** when someone starts saying something meaningful, not for "mm-hm" or laughter.
* If two people talk at once, mark the one who **holds the floor**.
* If you make a mistake, press the correct key straight away. A few seconds of error barely
  changes the results.
* **Never** press invite repeatedly at a silent participant. The robot is designed never to insist.

---

## Part 3 — After the study: get the results
```powershell
python -m nuru analyze logs/ --questionnaire data/questionnaire.csv
```
This creates the `results/` folder containing:
* `report.md`: table of all measures per condition, with statistical tests
* `summary_by_block.csv`: one row per group × condition, to open in Excel, SPSS or R
* `gini_by_condition.png`: figure for the report

---

## Part 4 — Phase 1: recording co-design clips
```powershell
python -m nuru clips --list         # see the 12 clips
python -m nuru clips V00 V03 V08    # play selected clips
python -m nuru clips all            # play all, with 4 s between clips
```
Each clip shows the robot inviting the person in seat B (change this with `--target C`). Film the robot
from that seat's viewpoint. The clip start and end times are saved in `logs/nuru_CLIPS_*.csv` to help
you cut the video.

---

## Part 5 — Practising without a robot
Add `--mock` to any command, for example `python -m nuru panel --mock`. Robot actions are then printed
in the terminal instead of performed. This is useful for training new operators.

## Part 6 — Optional: automatic speech detection with microphones
If each participant wears a microphone connected to one multichannel USB audio interface:
```powershell
python -m sounddevice                                   # find the interface's device number
python -m nuru panel --vad --device 5 --channels 0,1,2,3
```
Nuru then detects who is speaking automatically. The operator only corrects mistakes using the same
keys, which is the Wizard-of-Oz fallback. Live microphone levels appear on each seat tile.
