"""Nuru command line.   python -m nuru <command> [options]

  check       test the connection to Furhat and every capability Nuru uses
  panel       start the operator web panel (main tool for running sessions)
  cli         keyboard-only operator console (backup to the panel)
  demo        30-second live demonstration of C1, C2 and C3 invitations
  clips       Phase 1: play co-design stimulus clips (--list to see them)
  calibrate   check where the robot looks for each seat / calibrate from faces
  plan        print the counterbalanced schedule (condition order + topics)
  analyze     compute participation measures from logs -> results/
  simulate    synthetic groups: pipeline test and power analysis

Add --mock to any robot command to run without the robot.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import ROOT, load_config, save_seats
from .robot import make_robot


def _robot(args, cfg):
    if getattr(args, "host", None):
        cfg.host = args.host
    return make_robot(cfg, mock=getattr(args, "mock", False))


# --------------------------------------------------------------------- check
def cmd_check(args, cfg):
    from .cues import CUSTOM_GESTURES
    robot = _robot(args, cfg)
    ok_all = True

    def report(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail and not ok else ""))

    print(f"Checking Furhat at {cfg.host}:{cfg.port}")
    report("Remote API reachable", robot.ping(), "is the Remote API skill running on the robot?")
    if args.mock:
        return 0
    voices = {v["name"] for v in robot.voices_available()}
    for lang, v in cfg.voices.items():
        report(f"voice {lang} = {v}", v in voices)
    gestures = robot.gestures_available()
    report("built-in Nod gesture", "Nod" in gestures)
    report("custom gesture NuruSoftNod", robot.gesture(CUSTOM_GESTURES["NuruSoftNod"], blocking=True))
    report("gaze to group centre", robot.attend_location(cfg.group_centre))
    report("speech (en)", robot.say("Nuru check.", "en"))
    report("speech (sw)", robot.say("Habari.", "sw"))
    users = robot.users()
    print(f"  [info] robot currently detects {len(users)} face(s)")
    print("\nAll good - ready for a session." if ok_all else "\nSome checks failed - see docs/TROUBLESHOOTING.md")
    return 0 if ok_all else 1


# --------------------------------------------------------------------- panel
def cmd_panel(args, cfg):
    from .panel import serve
    robot = _robot(args, cfg)
    vad_opts = None
    if args.vad:
        vad_opts = {"device": args.device,
                    "channels": [int(c) for c in args.channels.split(",")] if args.channels else None}
    serve(cfg, robot, host=args.bind, port=args.port, vad_opts=vad_opts)


# ----------------------------------------------------------------------- cli
CLI_HELP = """Commands:  A-D speaker started | . silence | i invite | i A invite seat A
           c1/c2/c3 [T1] start condition | e end condition | n next question
           g greeting | auto on/off | s stats | stop | q quit"""


def cmd_cli(args, cfg):
    from .controller import NuruController
    robot = _robot(args, cfg)
    ctrl = NuruController(cfg, robot, group=args.group.upper())
    print(CLI_HELP)
    print(f"Logging to {ctrl.log.path}")
    try:
        while True:
            parts = input("> ").strip().split()
            if not parts:
                continue
            cmd, rest = parts[0].lower(), parts[1:]
            if cmd.upper() in cfg.seats:
                ctrl.speak(cmd.upper())
            elif cmd == ".":
                ctrl.silence()
            elif cmd == "i":
                seat = ctrl.invite(rest[0].upper() if rest else None)
                print(f"  inviting {seat}" if seat else "  (no condition running, or already inviting)")
            elif cmd in cfg.profiles:
                ctrl.start_condition(cmd, rest[0].upper() if rest else None)
                print(f"  {cfg.profiles[cmd]['label']}")
            elif cmd == "e":
                m = ctrl.end_condition()
                print(f"  ended: Gini time {m['gini_time']}" if m else "  no condition running")
            elif cmd == "n":
                print("  Q:", ctrl.next_question())
            elif cmd == "g":
                ctrl.greet()
            elif cmd == "auto":
                ctrl.set_auto(bool(rest) and rest[0] == "on")
            elif cmd == "s":
                m = ctrl.tracker.metrics()
                for s in cfg.seats:
                    print(f"  {s}: {m['speaking_time_s'][s]:6.1f}s  {m['turns'][s]} turns  invited {m['invitations'][s]}")
                print(f"  Gini time {m['gini_time']:.3f}  turns {m['gini_turns']:.3f}  response rate {m['response_rate']}")
            elif cmd == "stop":
                ctrl.stop_robot()
            elif cmd == "q":
                break
            else:
                print(CLI_HELP)
    except (KeyboardInterrupt, EOFError):
        pass
    ctrl.shutdown()
    print(f"Saved {ctrl.log.path}")


# ---------------------------------------------------------------------- demo
def cmd_demo(args, cfg):
    """Shows the three conditions inviting the same (pretend) quiet seat."""
    from .controller import NuruController
    robot = _robot(args, cfg)
    ctrl = NuruController(cfg, robot, group="DEMO")
    seats = list(cfg.seats)
    quiet = seats[-1]
    script = [("A dominant speaker talks", seats[0], 3.0), ("then another", seats[1], 2.0)]
    try:
        for cond in ("c1", "c2", "c3"):
            print(f"\n=== {cfg.profiles[cond]['label']}")
            ctrl.start_condition(cond, None, intro=False)
            robot.say(f"Condition {cond[1]}.", "en")
            for what, seat, dur in script:
                print(f"  {what}: seat {seat}")
                ctrl.speak(seat)
                time.sleep(dur)
            ctrl.silence()
            time.sleep(1.0)
            seat = ctrl.invite()
            print(f"  robot invites seat {seat} (policy: {cfg.profiles[cond]['policy']})")
            while ctrl._inviting:
                time.sleep(0.2)
            ctrl.end_condition()
            time.sleep(1.0)
        robot.attend_location(cfg.group_centre)
    finally:
        ctrl.shutdown()
    print(f"\nDemo log: {ctrl.log.path}")


# --------------------------------------------------------------------- clips
def cmd_clips(args, cfg):
    from .clips import CLIPS, list_clips, run_clips
    if args.list or not args.ids:
        list_clips()
        return
    ids = list(CLIPS) if args.ids == ["all"] else [i.upper() for i in args.ids]
    unknown = [i for i in ids if i not in CLIPS]
    if unknown:
        sys.exit(f"Unknown clip(s): {unknown}. Use --list.")
    run_clips(cfg, _robot(args, cfg), ids, target=args.target, gap_s=args.gap)


# ----------------------------------------------------------------- calibrate
def cmd_calibrate(args, cfg):
    from .controller import seats_from_users
    robot = _robot(args, cfg)
    if args.faces:
        print("Ask one person to sit in each seat, facing the robot...")
        time.sleep(2)
        seats = seats_from_users(robot.users(), cfg.seats)
        if not seats:
            sys.exit(f"Robot sees {len(robot.users())} face(s); need exactly {len(cfg.seats)}.")
        save_seats(cfg, seats)
        print("Saved seats to config/nuru.json:")
        for s, p in seats.items():
            print(f"  {s}: {p}")
    print("Robot will now look at each seat and say its letter. Check it looks at the right person.")
    for s, p in cfg.seats.items():
        print(f"  {s} -> {p}")
        robot.attend_location(p)
        robot.say(s, "en")
        time.sleep(1.2)
    robot.attend_location(cfg.group_centre)
    print("If a seat is off, edit its x value in config/nuru.json (+/-0.1 m) and re-run.")


# ---------------------------------------------------------------- plan/analyze/sim
def cmd_plan(args, cfg):
    from .design import print_schedule
    print_schedule(args.groups)


def cmd_analyze(args, cfg):
    from .analysis import analyze
    res = analyze(Path(args.logs), Path(args.out), Path(args.questionnaire) if args.questionnaire else None)
    print(Path(res["report"]).read_text(encoding="utf-8"))
    print(f"Tables and figure written to {args.out}/")


def cmd_simulate(args, cfg):
    from . import simulate
    pr = tuple(float(x) for x in args.p_respond.split(","))
    if args.power:
        simulate.power(args.groups, pr, reps=args.reps)
    else:
        print(f"Simulating {args.groups} synthetic groups (NOT real data)")
        simulate.run(args.groups, pr, seed=args.seed, write_logs=Path(args.write_logs) if args.write_logs else None)
        if args.write_logs:
            print(f"Synthetic logs in {args.write_logs}/ -> try: python -m nuru analyze {args.write_logs} --out results_sim")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m nuru", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to nuru.json (default config/nuru.json)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def robot_args(sp):
        sp.add_argument("--mock", action="store_true", help="no robot: print actions instead")
        sp.add_argument("--host", help="override robot IP from config")
        return sp

    robot_args(sub.add_parser("check"))
    sp = robot_args(sub.add_parser("panel"))
    sp.add_argument("--port", type=int, default=5000)
    sp.add_argument("--bind", default="127.0.0.1", help="0.0.0.0 to open the panel from a tablet")
    sp.add_argument("--vad", action="store_true", help="detect speech from per-seat microphones")
    sp.add_argument("--device", type=int, help="audio input device index (python -m sounddevice)")
    sp.add_argument("--channels", help="input channels for seats, e.g. 0,1,2,3")
    sp = robot_args(sub.add_parser("cli"))
    sp.add_argument("--group", default="G00")
    robot_args(sub.add_parser("demo"))
    sp = robot_args(sub.add_parser("clips"))
    sp.add_argument("ids", nargs="*", help="clip ids (V00..V11) or 'all'")
    sp.add_argument("--list", action="store_true")
    sp.add_argument("--target", help="seat the robot invites in the clips (default B)")
    sp.add_argument("--gap", type=float, default=4.0, help="seconds between clips")
    sp = robot_args(sub.add_parser("calibrate"))
    sp.add_argument("--faces", action="store_true", help="set seat positions from detected faces")
    sp = sub.add_parser("plan")
    sp.add_argument("--groups", type=int, default=10)
    sp = sub.add_parser("analyze")
    sp.add_argument("logs", nargs="?", default=str(ROOT / "logs"))
    sp.add_argument("--out", default=str(ROOT / "results"))
    sp.add_argument("--questionnaire", help="CSV: group,condition,seat,item,score")
    sp = sub.add_parser("simulate")
    sp.add_argument("--groups", type=int, default=10)
    sp.add_argument("--p-respond", default="0.5,0.5,0.5", help="P(answer invitation) for C1,C2,C3")
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--write-logs", help="directory to write synthetic logs to")
    sp.add_argument("--power", action="store_true")
    sp.add_argument("--reps", type=int, default=300)

    for stream in (sys.stdout, sys.stderr):   # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    fn = {"check": cmd_check, "panel": cmd_panel, "cli": cmd_cli, "demo": cmd_demo, "clips": cmd_clips,
          "calibrate": cmd_calibrate, "plan": cmd_plan, "analyze": cmd_analyze, "simulate": cmd_simulate}[args.cmd]
    return fn(args, cfg) or 0


if __name__ == "__main__":
    sys.exit(main())
