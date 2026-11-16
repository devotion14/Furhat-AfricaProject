"""Loading of the JSON configuration files in ./config."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

Point = tuple[float, float, float]


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Keys starting with "_" are comments.
    return _strip_comments(data)


def _strip_comments(obj):
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


@dataclass
class Config:
    host: str = "localhost"
    port: int = 54321
    timeout_s: float = 20.0
    voices: dict = field(default_factory=lambda: {"en": "AsiliaNeural", "sw": "ZuriNeural"})
    seats: dict[str, Point] = field(default_factory=dict)
    group_centre: Point = (0.0, 0.0, 1.2)
    moderation: dict = field(default_factory=dict)
    profiles: dict = field(default_factory=dict)
    tasks: dict = field(default_factory=dict)
    log_dir: Path = ROOT / "logs"
    source: Path | None = None

    # moderation shortcuts ---------------------------------------------------
    def mod(self, key: str, default=None):
        return self.moderation.get(key, default)

    def profile(self, condition: str) -> dict:
        return copy.deepcopy(self.profiles[condition.lower()])

    @property
    def seat_ids(self) -> list[str]:
        return list(self.seats.keys())


def load_config(path: str | Path | None = None,
                profiles_path: str | Path | None = None,
                tasks_path: str | Path | None = None) -> Config:
    path = Path(path) if path else CONFIG_DIR / "nuru.json"
    raw = _load_json(path)
    profiles = _load_json(Path(profiles_path) if profiles_path else CONFIG_DIR / "profiles.json")
    tasks = _load_json(Path(tasks_path) if tasks_path else CONFIG_DIR / "tasks.json")

    robot = raw.get("robot", {})
    log_dir = Path(raw.get("log_dir", "logs"))
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir

    return Config(
        host=robot.get("host", "localhost"),
        port=int(robot.get("port", 54321)),
        timeout_s=float(robot.get("timeout_s", 20)),
        voices=raw.get("voices", {}),
        seats={k: tuple(map(float, v)) for k, v in raw["seats"].items()},
        group_centre=tuple(map(float, raw.get("group_centre", (0, 0, 1.2)))),
        moderation=raw.get("moderation", {}),
        profiles={k.lower(): v for k, v in profiles.items()},
        tasks=tasks,
        log_dir=log_dir,
        source=path,
    )


def save_seats(cfg: Config, seats: dict[str, Point]) -> None:
    """Write calibrated seat positions back into the main config file."""
    path = cfg.source or CONFIG_DIR / "nuru.json"
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    raw["seats"] = {k: [round(c, 3) for c in v] for k, v in seats.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
        f.write("\n")
    cfg.seats = dict(seats)
