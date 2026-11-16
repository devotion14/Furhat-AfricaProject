"""Offline analysis of Nuru event logs -> the measures named in the abstract.

Usage
    python -m nuru analyze logs/                      # all sessions
    python -m nuru analyze logs/ --questionnaire data/questionnaire.csv

Measures are recomputed from the raw speech events (not from the live
counters), so a crashed or restarted session is still analysable.

Per group x condition
    gini_time, gini_turns              participation equity (lower = more equal)
    least_share_time / _turns          share taken by the least active member
    quiet_share_time                   share taken by self-reported quiet members (RQ3)
    invitations, response_rate, mean_latency_s      engagement with robot cues
    voluntary_turns                    turns not answering an invitation
Per condition
    mean / sd of the above, plus Likert items if a questionnaire file is given
Inference (if >= 2 complete groups)
    Friedman test across C1/C2/C3, and the two planned contrasts
    C1 vs C2 (adaptivity) and C2 vs C3 (cultural adaptation), Wilcoxon signed-rank.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .participation import gini

KEY_MEASURES = ["gini_time", "gini_turns", "least_share_time", "least_share_turns",
                "quiet_share_time", "response_rate", "mean_latency_s"]


def read_log(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _value(row):
    v = row.get("value", "")
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v


def session_measures(rows: list[dict]) -> list[dict]:
    """Return one dict per (group, condition) block in this log."""
    seats, meta = [], {}
    blocks, cur = [], None
    speaker, t_start = None, 0.0

    def close_turn(t):
        nonlocal speaker
        if cur is not None and speaker is not None:
            cur["time"][speaker] += t - t_start
        speaker = None

    for r in rows:
        ev, t, seat = r["event"], float(r["t"]), r.get("seat", "")
        if ev == "session_start":
            v = _value(r)
            if isinstance(v, dict):
                seats = v.get("seats", seats)
        elif ev == "participant_meta":
            meta[seat] = r["value"]
        elif ev == "condition_start":
            v = _value(r)
            cur = {"group": r["group"], "condition": r["condition"],
                   "task": (v or {}).get("task") if isinstance(v, dict) else r.get("task"),
                   "t0": t, "t1": t,
                   "time": defaultdict(float), "turns": defaultdict(int),
                   "invites": defaultdict(int), "responses": defaultdict(int), "latency": []}
            speaker = None
        elif cur is None:
            continue
        elif ev == "speak_start":
            close_turn(t)
            speaker, t_start = seat, t
            cur["turns"][seat] += 1
        elif ev == "speak_stop":
            if seat == speaker:
                close_turn(t)
        elif ev == "invite":
            cur["invites"][seat] += 1
        elif ev == "response":
            cur["responses"][seat] += 1
            try:
                cur["latency"].append(float(r["value"]))
            except ValueError:
                pass
        elif ev == "condition_end":
            close_turn(t)
            cur["t1"] = t
            blocks.append(cur)
            cur = None
    if cur is not None:   # session ended without condition_end
        close_turn(float(rows[-1]["t"]))
        cur["t1"] = float(rows[-1]["t"])
        blocks.append(cur)

    out = []
    for b in blocks:
        ids = seats or sorted(set(b["time"]) | set(b["turns"]))
        tm = [b["time"][s] for s in ids]
        tn = [b["turns"][s] for s in ids]
        tot_t, tot_n = sum(tm), sum(tn)
        quiet = [s for s in ids if str(meta.get(s, "")).lower() == "quiet"]
        inv = sum(b["invites"].values())
        resp = sum(b["responses"].values())
        rec = {
            "group": b["group"], "condition": b["condition"], "task": b["task"],
            "duration_s": round(b["t1"] - b["t0"], 1),
            "total_speech_s": round(tot_t, 1), "total_turns": tot_n,
            "gini_time": round(gini(tm), 4), "gini_turns": round(gini(tn), 4),
            "least_share_time": round(min(tm) / tot_t, 4) if tot_t else None,
            "least_share_turns": round(min(tn) / tot_n, 4) if tot_n else None,
            "quiet_seats": ",".join(quiet),
            "quiet_share_time": round(sum(b["time"][s] for s in quiet) / tot_t, 4) if quiet and tot_t else None,
            "quiet_response_rate": (round(sum(b["responses"][s] for s in quiet) / sum(b["invites"][s] for s in quiet), 4)
                                    if quiet and sum(b["invites"][s] for s in quiet) else None),
            "invitations": inv, "responses": resp,
            "response_rate": round(resp / inv, 4) if inv else None,
            "mean_latency_s": round(mean(b["latency"]), 2) if b["latency"] else None,
            "voluntary_turns": tot_n - resp,
        }
        for s in ids:
            rec[f"time_{s}"] = round(b["time"][s], 1)
            rec[f"turns_{s}"] = b["turns"][s]
        out.append(rec)
    return out


def _stats(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None, None, 0
    return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def inferential(per_block: list[dict], key: str) -> dict:
    """Friedman + planned Wilcoxon contrasts over groups that have all 3 conditions."""
    try:
        from scipy import stats
    except ImportError:
        return {"note": "scipy not installed"}
    by_group = defaultdict(dict)
    for r in per_block:
        if r.get(key) is not None:
            by_group[r["group"]][r["condition"]] = r[key]
    complete = {g: d for g, d in by_group.items() if all(c in d for c in ("c1", "c2", "c3"))}
    n = len(complete)
    res = {"n_groups": n}
    if n < 2:
        res["note"] = "need >= 2 groups with all three conditions"
        return res
    c1 = [d["c1"] for d in complete.values()]
    c2 = [d["c2"] for d in complete.values()]
    c3 = [d["c3"] for d in complete.values()]
    try:
        if n >= 3:
            f = stats.friedmanchisquare(c1, c2, c3)
            res["friedman_chi2"], res["friedman_p"] = round(float(f.statistic), 3), round(float(f.pvalue), 4)
        for name, a, b in (("c1_vs_c2", c1, c2), ("c2_vs_c3", c2, c3)):
            if any(x != y for x, y in zip(a, b)):
                w = stats.wilcoxon(a, b)
                res[f"{name}_W"], res[f"{name}_p"] = round(float(w.statistic), 3), round(float(w.pvalue), 4)
    except ValueError as e:
        res["note"] = str(e)
    return res


def read_questionnaire(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("score", "").strip()]


def analyze(log_dir: Path, out_dir: Path, questionnaire: Path | None = None, plot: bool = True) -> dict:
    files = sorted(Path(log_dir).glob("nuru_*.csv"))
    per_block = []
    for f in files:
        rows = read_log(f)
        if rows:
            for rec in session_measures(rows):
                rec["log_file"] = f.name
                per_block.append(rec)
    out_dir.mkdir(parents=True, exist_ok=True)

    # per block table
    if per_block:
        keys = list(dict.fromkeys(k for r in per_block for k in r))
        with open(out_dir / "summary_by_block.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(per_block)

    by_cond = defaultdict(list)
    for r in per_block:
        by_cond[r["condition"]].append(r)
    conds = [c for c in ("c1", "c2", "c3") if c in by_cond]

    lines = ["# Nuru results summary", "",
             f"Logs analysed: {len(files)} file(s), {len(per_block)} condition block(s).", "",
             "## Participation equity and engagement (mean ± sd over groups)", "",
             "| Measure | " + " | ".join(c.upper() for c in conds) + " |",
             "|---|" + "---|" * len(conds)]
    for k in KEY_MEASURES + ["invitations", "voluntary_turns", "total_speech_s"]:
        cells = []
        for c in conds:
            m, s, n = _stats(by_cond[c], k)
            cells.append("–" if m is None else f"{m:.3f} ± {s:.3f} (n={n})")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")

    lines += ["", "## Inferential tests (groups with all three conditions)", "",
              "Planned contrasts: **C1 vs C2** = effect of adaptivity; **C2 vs C3** = effect of cultural adaptation.", ""]
    tests = {k: inferential(per_block, k) for k in ("gini_time", "gini_turns", "least_share_time", "quiet_share_time")}
    for k, v in tests.items():
        lines.append(f"- **{k}**: " + ", ".join(f"{a}={b}" for a, b in v.items()))

    if questionnaire and Path(questionnaire).exists():
        q = read_questionnaire(Path(questionnaire))
        items = sorted({r["item"] for r in q})
        lines += ["", "## Questionnaire (Likert 1–7, mean ± sd)", "",
                  "| Item | " + " | ".join(c.upper() for c in ("c1", "c2", "c3")) + " |", "|---|---|---|---|"]
        for it in items:
            cells = []
            for c in ("c1", "c2", "c3"):
                vals = [float(r["score"]) for r in q if r["item"] == it and r["condition"].lower() == c]
                cells.append(f"{mean(vals):.2f} ± {stdev(vals) if len(vals) > 1 else 0:.2f} (n={len(vals)})" if vals else "–")
            lines.append(f"| {it} | " + " | ".join(cells) + " |")

    lines += ["", "_Lower Gini = more equal participation. Shares are proportions (0–1)._", ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    if plot and per_block:
        _plot(per_block, out_dir / "gini_by_condition.png")
    return {"blocks": per_block, "tests": tests, "report": str(out_dir / "report.md")}


def _plot(per_block, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    order = ["c1", "c2", "c3"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, key, title in ((axes[0], "gini_time", "Gini of speaking time"),
                           (axes[1], "gini_turns", "Gini of turns")):
        groups = defaultdict(dict)
        for r in per_block:
            groups[r["group"]][r["condition"]] = r[key]
        for g, d in groups.items():
            xs = [i for i, c in enumerate(order) if c in d]
            ax.plot(xs, [d[order[i]] for i in xs], "-o", color="#999", alpha=0.5, lw=1, ms=3)
        means = [mean([r[key] for r in per_block if r["condition"] == c] or [float("nan")]) for c in order]
        ax.plot(range(3), means, "-o", color="#c0392b", lw=2.5, ms=7, label="mean")
        ax.set_xticks(range(3), ["C1\nnon-adaptive", "C2\nadaptive-conv.", "C3\nadaptive-Nuru"])
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Gini (lower = more equal)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
