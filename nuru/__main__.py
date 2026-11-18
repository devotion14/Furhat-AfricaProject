"""Nuru command line.   python -m nuru <command> [options]

  check       test the connection to Furhat and every capability Nuru uses
  panel       start the operator web panel (main tool for running sessions)
  describe    the robot says hello and comments on each person's expression and
              clothing colour, tailoring what it says (social-features lab demo)
  session     fully automatic session: Furhat's camera+mic detect who speaks,
              the robot greets, runs C1/C2/C3 with their topics and invites by itself
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


# ------------------------------------------------- automatic speech detection
def _avsd_opts(args, cfg) -> dict:
    """Options for nuru.avsd.AVSDRunner from --camera / --mic / --threshold / --motion."""
    furhat_cam = args.camera == "furhat"
    return {"camera": 0 if furhat_cam else int(args.camera),
            "furhat_ip": cfg.host if furhat_cam or args.mic == "furhat" else None,
            "furhat_audio": args.mic == "furhat",
            "mic_device": None if args.mic in ("furhat", "default") else int(args.mic),
            "threshold_db": args.threshold, "min_motion": args.motion}


def avsd_args(sp):
    sp.add_argument("--camera", default="furhat",
                    help="'furhat' (robot camera, default) or a webcam number, e.g. 0")
    sp.add_argument("--mic", default="furhat",
                    help="'furhat' (robot mic, default), 'default' (laptop mic) or a device number")
    sp.add_argument("--threshold", type=float, default=9.0, help="mic sensitivity dB (lower = more sensitive)")
    sp.add_argument("--motion", type=float, default=0.004, help="mouth-motion threshold (lower = more sensitive)")
    return sp


# --------------------------------------------------------------------- panel
def cmd_panel(args, cfg):
    from .panel import serve
    robot = _robot(args, cfg)
    vad_opts = None
    if args.avsd:
        vad_opts = {"kind": "avsd", "auto": args.auto, **_avsd_opts(args, cfg)}
    elif args.vad:
        vad_opts = {"device": args.device,
                    "channels": [int(c) for c in args.channels.split(",")] if args.channels else None,
                    "auto": args.auto}
    serve(cfg, robot, host=args.bind, port=args.port, vad_opts=vad_opts)


# ------------------------------------------------------------------- session
def _wait_robot(ctrl, timeout=90.0):
    """Wait until the robot has finished everything queued (speech, invitations)."""
    t0 = time.time()
    time.sleep(0.3)
    while time.time() - t0 < timeout:
        if ctrl._q.empty() and ctrl._inviting is None and ctrl.status == "idle":
            return
        time.sleep(0.2)


def _introduce(ctrl, robot, det, cfg, look_s=3.0):
    """After hello: look at each person, mention their top's colour and facial
    expression, and tailor the line (e.g. reassurance for a serious face)."""
    from .avsd import describe_sentence
    print("Looking at the participants...")
    time.sleep(look_s)                       # collect a few seconds of observations
    for seat, point in cfg.seats.items():
        robot.attend_location(point)
        time.sleep(0.8)                      # let the robot settle its gaze on the person
        obs = det.observation(seat)
        text = describe_sentence(obs)
        print(f"  {seat}: {obs or 'not seen'}  ->  \"{text}\"")
        ctrl.log.write("social_features", seat, value=obs or {})
        ctrl.log.write("robot_say", seat, text, lang="en")
        robot.say(text, "en", blocking=True)
    robot.attend_location(cfg.group_centre)


def _start_detector(args, cfg, ctrl, on_change=None):
    """Start camera+mic speaker detection. If it cannot start, the session still
    runs: presence comes from Furhat's own face tracking and invitations fall
    back to the timed rule (see _check_detector)."""
    from .avsd import AVSDRunner
    det = AVSDRunner(ctrl, on_change=on_change, **_avsd_opts(args, cfg))
    det.start()
    print("Starting the robot's camera and microphone (first time can take ~20 s)...")
    time.sleep(5)
    _check_detector(ctrl, det)
    return det


def _check_detector(ctrl, det):
    """Report (once) that speaker detection is down and switch to the default
    behaviour: no speaker info, so the robot keeps moderating with timed
    invitations (auto-invite after a lull) instead of freezing."""
    if det.error and not getattr(det, "_reported", False):
        det._reported = True
        print(f"\n  WARNING: {det.error}\n  Continuing without speaker detection: "
              "the robot invites on a timer.")
        ctrl.log.write("detector_failed", "", det.error)
        ctrl.silence()
        return False
    return not det.error


def _wait_for_people(robot, det, hold_s=2.0, mock=False):
    """SENSE step that starts the interaction: block until someone has been in
    front of the robot for hold_s seconds. Uses our camera detector, or
    Furhat's own user tracking if the detector is not running."""
    print("Waiting for someone to sit down in front of the robot...  (Ctrl+C to quit)")
    since = None
    while True:
        if not det.error:
            if det.present_s() >= hold_s:
                return "camera"
        elif mock:
            return "mock"
        elif robot.users():
            since = since or time.time()
            if time.time() - since >= hold_s:
                return "furhat"
        else:
            since = None
        time.sleep(0.2)


def _start_when_present(args, robot, det, ctrl):
    if args.start_now:
        print("Starting now (--start-now).")
        ctrl.log.write("session_trigger", "", "manual")
        return
    how = _wait_for_people(robot, det, mock=args.mock)
    print(f"Person detected ({how}) - starting.")
    ctrl.log.write("session_trigger", "", how)


def cmd_describe(args, cfg):
    """Lab demo: hello + tailored comment on each person's expression and clothing."""
    from .controller import NuruController
    robot = _robot(args, cfg)
    if not args.mock and not robot.ping():
        sys.exit(f"Robot at {cfg.host}:{cfg.port} not reachable - start the Remote API skill on the robot.")
    ctrl = NuruController(cfg, robot, group=args.group.upper())
    det = _start_detector(args, cfg, ctrl)
    try:
        _start_when_present(args, robot, det, ctrl)
        ctrl.greet()
        _wait_robot(ctrl)
        _introduce(ctrl, robot, det, cfg)
    finally:
        det.stop()
        ctrl.shutdown()
    print(f"Saved {ctrl.log.path}")


def cmd_session(args, cfg):
    """Fully automatic session. The operator only starts it and presses Enter
    between conditions (so participants can fill in the questionnaire)."""
    from .controller import NuruController

    robot = _robot(args, cfg)
    if not args.mock and not robot.ping():
        sys.exit(f"Robot at {cfg.host}:{cfg.port} not reachable - start the Remote API skill on the robot.")
    conds = [c.lower() for c in args.conditions]
    unknown = [c for c in conds if c not in cfg.profiles]
    if unknown:
        sys.exit(f"Unknown condition(s) {unknown}; choose from {list(cfg.profiles)}")
    all_tasks = list(cfg.tasks["tasks"])
    if args.tasks and any(t.upper() not in cfg.tasks["tasks"] for t in args.tasks):
        sys.exit(f"Unknown task in {args.tasks}; choose from {all_tasks}")

    ctrl = NuruController(cfg, robot, group=args.group.upper())

    def on_change(seat):
        if not args.quiet:
            print(f"    {time.strftime('%H:%M:%S')}  {('speaking: ' + seat) if seat else 'silence'}")

    det = _start_detector(args, cfg, ctrl, on_change)

    ctrl.set_auto(True)
    try:
        _start_when_present(args, robot, det, ctrl)
    except KeyboardInterrupt:
        det.stop()
        ctrl.shutdown()
        return 1
    print(f"Group {args.group.upper()}  conditions {' -> '.join(c.upper() for c in conds)}  "
          f"{args.minutes:g} min each.  Ctrl+C = emergency stop.\nLog: {ctrl.log.path}")
    try:
        ctrl.greet()
        _wait_robot(ctrl)
        if not args.no_describe:
            _introduce(ctrl, robot, det, cfg)
        for i, cond in enumerate(conds):
            task = (args.tasks[i].upper() if args.tasks and i < len(args.tasks)
                    else all_tasks[i % len(all_tasks)])
            questions = cfg.tasks["tasks"][task]["questions"]
            n_q = max(1, min(len(questions), args.questions))
            print(f"\n=== {cond.upper()}: {cfg.profiles[cond].get('label', '')}  |  task {task}: "
                  f"{cfg.tasks['tasks'][task]['title']}")
            ctrl.start_condition(cond, task)
            _wait_robot(ctrl)
            ctrl.next_question()
            _wait_robot(ctrl)
            dur = args.minutes * 60
            t0, asked = time.time(), 1
            while time.time() - t0 < dur:
                time.sleep(0.5)
                _check_detector(ctrl, det)
                due = asked < n_q and time.time() - t0 >= asked * dur / n_q
                late = due and time.time() - t0 >= asked * dur / n_q + 20
                # next question when nobody is speaking (or 20 s late at the latest)
                if due and ctrl._inviting is None and (ctrl.tracker.current is None or late):
                    ctrl.next_question()
                    asked += 1
            _wait_robot(ctrl)
            m = ctrl.end_condition() or {}
            print(f"--- {cond.upper()} done:  Gini(time) {m.get('gini_time')}  "
                  f"response rate {m.get('response_rate')}")
            for s in cfg.seats:
                print(f"      {s}: {m.get('speaking_time_s', {}).get(s, 0):6.1f} s  "
                      f"{m.get('turns', {}).get(s, 0)} turns  invited {m.get('invitations', {}).get(s, 0)}")
            if i < len(conds) - 1:
                robot.attend_location(cfg.group_centre)
                if args.no_pause:
                    time.sleep(args.break_s)
                else:
                    input("\n  Questionnaire time. Press Enter to start the next condition... ")
        ctrl.close()
        _wait_robot(ctrl)
    except KeyboardInterrupt:
        print("\nStopped by operator.")
        ctrl.stop_robot()
    finally:
        det.stop()
        ctrl.shutdown()
    print(f"\nSaved {ctrl.log.path}\nAnalyse with:  python -m nuru analyze logs/")


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
    sp.add_argument("--avsd", action="store_true",
                    help="detect who speaks with Furhat's camera + mic (see --camera/--mic)")
    sp.add_argument("--auto", action="store_true", help="turn automatic invitations on at session start")
    avsd_args(sp)
    sp = avsd_args(robot_args(sub.add_parser("session")))
    sp.add_argument("--group", default="G00", help="group id for the log, e.g. G03")
    sp.add_argument("--conditions", nargs="+", default=["c1", "c2", "c3"], help="order, e.g. c3 c1 c2")
    sp.add_argument("--tasks", nargs="+", help="task id per condition, e.g. T2 T3 T1 (default: in order)")
    sp.add_argument("--minutes", type=float, default=3.0, help="length of each condition")
    sp.add_argument("--questions", type=int, default=2, help="discussion questions asked per condition")
    sp.add_argument("--no-pause", action="store_true", help="don't wait for Enter between conditions")
    sp.add_argument("--break-s", type=float, default=10.0, help="break length with --no-pause")
    sp.add_argument("--quiet", action="store_true", help="don't print every speaker change")
    sp.add_argument("--no-describe", action="store_true",
                    help="skip commenting on expression/clothing after hello")
    sp.add_argument("--start-now", action="store_true",
                    help="start immediately instead of waiting for a person to be detected")
    sp = avsd_args(robot_args(sub.add_parser("describe")))
    sp.add_argument("--group", default="G00")
    sp.add_argument("--start-now", action="store_true",
                    help="start immediately instead of waiting for a person to be detected")
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
    fn = {"check": cmd_check, "panel": cmd_panel, "session": cmd_session, "describe": cmd_describe, "cli": cmd_cli, "demo": cmd_demo, "clips": cmd_clips,
          "calibrate": cmd_calibrate, "plan": cmd_plan, "analyze": cmd_analyze, "simulate": cmd_simulate}[args.cmd]
    return fn(args, cfg) or 0


if __name__ == "__main__":
    sys.exit(main())
