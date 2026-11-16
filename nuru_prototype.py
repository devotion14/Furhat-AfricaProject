"""
Nuru prototype - Wizard-of-Oz participation tracker + robot behaviour (C1/C2/C3)

The operator watches the group and types who is speaking. The script keeps
track of speaking time and turns per seat, and when you press "i" the robot
invites the next person according to the active condition.

Setup:
    pip install furhat-remote-api
    Start the virtual (or real) Furhat with the Remote API skill running.
    Set ROBOT_IP below ("localhost" if the robot runs on this laptop).

Commands (type and press Enter):
    A / B / C / D   that person has started speaking
    .               silence (current speaker stopped)
    i               robot invites the next person (uses the active condition)
    c1 / c2 / c3    switch condition
    s               show stats (time, turns, Gini)
    q               quit and save the log
"""

import csv
import time
from datetime import datetime

from furhat_remote_api import FurhatRemoteAPI

ROBOT_IP = "localhost"

# Seat positions in front of the robot, in metres: "x,y,z".
# x = left/right, y = up/down, z = distance from robot.
# CALIBRATE THESE: use the GAZE pad or the Dashboard to see which x values
# make the robot look at each seat, and adjust.
SEATS = {
    "A": "-0.6,0.0,1.2",
    "B": "-0.2,0.0,1.2",
    "C": "0.2,0.0,1.2",
    "D": "0.6,0.0,1.2",
}
GROUP_CENTRE = "0.0,0.0,1.2"

# Behaviour profiles. C3 values are placeholders until the co-design
# sessions decide them.
PROFILES = {
    "c1": {  # Non-adaptive: cycles through everyone in order
        "policy": "round_robin",
        "group_glance": False,
        "gaze_hold": 2.0,
        "nod": False,
        "prompt": "What do you think?",
        "wait_after": 2.0,
    },
    "c2": {  # Adaptive, conventional cues: direct, held gaze at quietest
        "policy": "least_active",
        "group_glance": False,
        "gaze_hold": 3.0,
        "nod": False,
        "prompt": "What do you think?",
        "wait_after": 2.0,
    },
    "c3": {  # Adaptive, Nuru cues: group first, shorter gaze, nod, longer wait
        "policy": "least_active",
        "group_glance": True,
        "gaze_hold": 1.5,
        "nod": True,
        "prompt": "Je, una maoni? What do you think?",
        "wait_after": 4.0,
    },
}

# How many recent invitations to avoid repeating (so no one is singled out)
NO_REPEAT_WINDOW = 2


class Tracker:
    def __init__(self, seats):
        self.time = {s: 0.0 for s in seats}
        self.turns = {s: 0 for s in seats}
        self.current = None
        self.started = None
        self.invited = []  # history of invited seats
        self.rr_index = 0

    def start(self, seat):
        self.stop()
        self.current = seat
        self.started = time.time()
        self.turns[seat] += 1

    def stop(self):
        if self.current is not None:
            self.time[self.current] += time.time() - self.started
        self.current = None
        self.started = None

    def gini(self):
        vals = sorted(self.time.values())
        n, total = len(vals), sum(vals)
        if total == 0:
            return 0.0
        cum = sum((i + 1) * v for i, v in enumerate(vals))
        return (2 * cum) / (n * total) - (n + 1) / n

    def choose(self, policy):
        seats = list(self.time.keys())
        if policy == "round_robin":
            seat = seats[self.rr_index % len(seats)]
            self.rr_index += 1
            return seat
        recent = set(self.invited[-NO_REPEAT_WINDOW:])
        candidates = [s for s in seats if s not in recent] or seats
        return min(candidates, key=lambda s: (self.time[s], self.turns[s]))


def invite(furhat, seat, p):
    if p["group_glance"]:
        furhat.attend(location=GROUP_CENTRE)
        time.sleep(1.0)
    furhat.attend(location=SEATS[seat])
    if p["nod"]:
        furhat.gesture(name="Nod")
    furhat.say(text=p["prompt"], blocking=True)
    time.sleep(p["gaze_hold"])
    furhat.attend(location=GROUP_CENTRE)
    time.sleep(p["wait_after"])


def main():
    furhat = FurhatRemoteAPI(ROBOT_IP)
    tracker = Tracker(SEATS)
    condition = "c1"
    log_name = f"nuru_log_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log_file = open(log_name, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["timestamp", "condition", "event", "seat"])

    def record(event, seat=""):
        log.writerow([f"{time.time():.3f}", condition, event, seat])
        log_file.flush()

    furhat.attend(location=GROUP_CENTRE)
    print(__doc__)
    print(f"Condition: {condition.upper()}   Logging to {log_name}")

    while True:
        cmd = input("> ").strip()
        key = cmd.upper()
        if key in SEATS:
            tracker.start(key)
            record("speak_start", key)
        elif cmd == ".":
            if tracker.current:
                record("speak_stop", tracker.current)
            tracker.stop()
        elif cmd.lower() == "i":
            tracker.stop()
            seat = tracker.choose(PROFILES[condition]["policy"])
            tracker.invited.append(seat)
            record("invite", seat)
            print(f"Inviting {seat} ({condition.upper()})")
            invite(furhat, seat, PROFILES[condition])
        elif cmd.lower() in PROFILES:
            tracker.stop()
            condition = cmd.lower()
            record("condition_change")
            print(f"Condition: {condition.upper()}")
        elif cmd.lower() == "s":
            for s in SEATS:
                print(f"  {s}: {tracker.time[s]:6.1f}s  {tracker.turns[s]} turns")
            print(f"  Gini (speaking time): {tracker.gini():.3f}")
        elif cmd.lower() == "q":
            tracker.stop()
            record("end")
            break
        else:
            print("Unknown command. Use A-D, '.', i, c1/c2/c3, s, q")

    log_file.close()
    print(f"Saved {log_name}. Final Gini: {tracker.gini():.3f}")


if __name__ == "__main__":
    main()
