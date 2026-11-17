"""Operator (Wizard-of-Oz) web panel.  python -m nuru panel  ->  http://localhost:5000

The operator marks who is speaking (click / keys 1-4, space = silence), starts
conditions, advances the discussion script and can trigger or automate
invitations. With --vad (per-seat microphones) or --avsd (Furhat's camera + mic),
speech is detected automatically and the operator only corrects mistakes
(WoZ fallback, as in the abstract).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from .config import save_seats
from .controller import NuruController, seats_from_users
from .design import schedule

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("nuru.panel")


def create_app(cfg, robot, vad_opts: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    st = {"ctrl": None, "vad": None, "vad_error": None}
    lock = threading.Lock()

    def ctrl() -> NuruController:
        c = st["ctrl"]
        if c is None:
            raise RuntimeError("no session")
        return c

    @app.errorhandler(RuntimeError)
    def no_session(e):
        return jsonify({"ok": False, "error": str(e)}), 409

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "panel.html")

    @app.get("/api/info")
    def info():
        return jsonify({"robot": f"{cfg.host}:{cfg.port}", "robot_online": robot.ping(),
                        "seats": list(cfg.seats), "conditions": {k: v.get("label") for k, v in cfg.profiles.items()},
                        "tasks": {k: v["title"] for k, v in cfg.tasks["tasks"].items()},
                        "vad": bool(vad_opts)})

    @app.get("/api/plan")
    def plan():
        g = request.args.get("group", "G01").upper()
        try:
            n = int(g.lstrip("G"))
        except ValueError:
            return jsonify({"blocks": []})
        p = schedule(max(n, 1))[n - 1] if n >= 1 else {"blocks": []}
        return jsonify(p)

    @app.get("/api/state")
    def state():
        c = st["ctrl"]
        if c is None:
            return jsonify({"session": False})
        s = c.state()
        s["session"] = True
        if st["vad"]:
            s["mic_db"] = st["vad"].levels()
            if getattr(st["vad"], "error", None):      # detector died after starting
                st["vad_error"] = st["vad"].error
        s["vad_error"] = st["vad_error"]
        return jsonify(s)

    @app.post("/api/session/start")
    def session_start():
        body = request.get_json(force=True, silent=True) or {}
        with lock:
            if st["ctrl"]:
                _end()
            c = NuruController(cfg, robot, group=body.get("group", "G00").upper())
            c.set_meta(body.get("meta", {}))
            st["ctrl"] = c
            if vad_opts:
                opts = dict(vad_opts)
                kind = opts.pop("kind", "vad")
                auto = opts.pop("auto", False)
                try:
                    if kind == "avsd":
                        from .avsd import AVSDRunner
                        st["vad"] = AVSDRunner(c, **opts)
                    else:
                        from .vad import VADRunner
                        st["vad"] = VADRunner(c, **opts)
                    st["vad"].start()
                    st["vad_error"] = None
                except Exception as e:  # mic/camera problems must not block the session
                    st["vad"], st["vad_error"] = None, f"Speech detection unavailable, use manual marking: {e}"
                if auto and st["vad"]:
                    c.set_auto(True)
        return jsonify({"ok": True, "log": str(c.log.path)})

    def _end():
        if st["vad"]:
            st["vad"].stop()
            st["vad"] = None
        res = st["ctrl"].results if st["ctrl"] else {}
        if st["ctrl"]:
            st["ctrl"].shutdown()
        st["ctrl"] = None
        return res

    @app.post("/api/session/end")
    def session_end():
        with lock:
            res = _end()
        return jsonify({"ok": True, "results": res})

    @app.post("/api/condition/start")
    def cond_start():
        b = request.get_json(force=True, silent=True) or {}
        ctrl().start_condition(b["condition"], b.get("task") or None, intro=b.get("intro", True))
        return jsonify({"ok": True})

    @app.post("/api/condition/end")
    def cond_end():
        return jsonify({"ok": True, "metrics": ctrl().end_condition()})

    @app.post("/api/speak/<seat>")
    def speak(seat):
        ctrl().speak(seat)
        return jsonify({"ok": True})

    @app.post("/api/silence")
    def silence():
        ctrl().silence()
        return jsonify({"ok": True})

    @app.post("/api/invite")
    def invite():
        b = request.get_json(force=True, silent=True) or {}
        seat = ctrl().invite(b.get("seat"), source="wizard")
        return jsonify({"ok": seat is not None, "seat": seat})

    @app.post("/api/question/next")
    def next_q():
        return jsonify({"ok": True, "question": ctrl().next_question()})

    @app.post("/api/greet")
    def greet():
        ctrl().greet()
        return jsonify({"ok": True})

    @app.post("/api/closing")
    def closing():
        ctrl().close()
        return jsonify({"ok": True})

    @app.post("/api/say")
    def say():
        b = request.get_json(force=True, silent=True) or {}
        ctrl().say(b.get("text", ""), b.get("lang", "en"))
        return jsonify({"ok": True})

    @app.post("/api/look/<target>")
    def look(target):
        ctrl().look(target)
        return jsonify({"ok": True})

    @app.post("/api/auto")
    def auto():
        b = request.get_json(force=True, silent=True) or {}
        ctrl().set_auto(bool(b.get("on")))
        return jsonify({"ok": True})

    @app.post("/api/note")
    def note():
        b = request.get_json(force=True, silent=True) or {}
        ctrl().note(b.get("text", ""))
        return jsonify({"ok": True})

    @app.post("/api/stop")
    def stop():
        ctrl().stop_robot()
        return jsonify({"ok": True})

    @app.post("/api/calibrate/sweep")
    def sweep():
        def run():
            for s, p in cfg.seats.items():
                robot.attend_location(p)
                robot.say(s, blocking=True)
                time.sleep(1.0)
            robot.attend_location(cfg.group_centre)
        threading.Thread(target=run, daemon=True).start()
        return jsonify({"ok": True})

    @app.post("/api/calibrate/faces")
    def faces():
        seats = seats_from_users(robot.users(), cfg.seats)
        if not seats:
            n = len(robot.users())
            return jsonify({"ok": False, "error": f"robot sees {n} face(s), need exactly {len(cfg.seats)}"})
        save_seats(cfg, seats)
        if st["ctrl"]:
            st["ctrl"].seats = cfg.seats
        return jsonify({"ok": True, "seats": seats})

    return app


def serve(cfg, robot, host="127.0.0.1", port=5000, vad_opts=None):
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app = create_app(cfg, robot, vad_opts)
    print(f"\n  Nuru operator panel:  http://{host if host != '0.0.0.0' else 'localhost'}:{port}\n"
          f"  Robot:                {cfg.host}:{cfg.port}  ({'online' if robot.ping() else 'NOT reachable'})\n")
    app.run(host=host, port=port, threaded=True, debug=False)
