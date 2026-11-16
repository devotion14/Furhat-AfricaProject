# Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `[FAIL] Remote API reachable` | The Remote API skill is not running, or a different network | Robot web UI (`http://<robot-ip>:8080`) → Skills → Remote API → Start. Check the laptop is on the same Wi-Fi. Run `curl http://<robot-ip>:54321/furhat`, which should answer "Hello World!". |
| Robot IP changed | DHCP | Update `"host"` in `config/nuru.json`, or pass `--host 192.168.x.y` |
| `[FAIL] voice …` | That voice is not installed on this robot | Run `python -c "from nuru.robot import FurhatRobot as F; print([v for v in F('192.168.0.105').voices_available() if 'KE' in v['language'] or 'TZ' in v['language']])"`, then choose another voice in `config/nuru.json` (for example `RehemaNeural` for sw-TZ or `ImaniNeural` for en-TZ) |
| Robot looks at the wrong person | Seats not calibrated | `python -m nuru calibrate --faces`, or adjust the x values by hand. If left and right are swapped, flip the sign of every x. |
| Robot looks over or under people's heads | y (height) is wrong | Change the second number of each seat (for example −0.2 if the robot sits above eye level) |
| Panel says "no session" | You have not pressed Start session | Press **Start session** first |
| Panel page does not load | Panel not running, or the port is taken | Run `python -m nuru panel --port 5050` and open http://localhost:5050 |
| Panel needed on a tablet | The panel only listens on this laptop | Run `python -m nuru panel --bind 0.0.0.0`, then open `http://<laptop-ip>:5000` on the tablet (same network) |
| Robot is mid-sentence and must stop | — | Press **Esc** or **STOP robot** |
| `--vad` shows "VAD unavailable" | Wrong device or channels | Run `python -m sounddevice` and pick the multichannel interface's index. Manual marking still works in the meantime. |
| VAD marks the wrong person | Microphones too close or gain too high | Lower the preamp gain, place each microphone 5–10 cm from its speaker's mouth; or raise `crosstalk_db` / `threshold_db` in `nuru/vad.py` |
| Custom gesture error "cannot be cast" | Wrong gesture class name | It must be `"class": "furhatos.gestures.Gesture"` (already correct in `cues.py`) |
| `UnicodeEncodeError` in the console | Windows code page | Fixed in the CLI; for other scripts, set `PYTHONIOENCODING=utf8` |
| Tests fail after editing profiles | C1 and C2 now differ in more than targeting | Intended: the test protects the study design. Keep C1 and C2 cue form identical. |
