"""Synthetic groups: a pipeline test and a power analysis. NOT evidence.

This simulates four-person discussions (one dominant, two average, one quiet
participant) under each condition's *invitation policy*, using the real
Tracker and policy code. Humans are modelled crudely:
    voluntary turn starts  ~ Bernoulli per second, scaled by talkativeness
    turn length            ~ exponential, longer for talkative people
    answering an invitation with probability p_respond[condition]

Because C2 and C3 share the participation model, any simulated C2/C3
difference comes ONLY from the p_respond you assume. That is exactly the
quantity the real study measures. Use the simulator to:
    1. test the logging -> analysis pipeline end to end  (--write-logs)
    2. estimate statistical power for a given number of groups (--power)

    python -m nuru simulate --groups 10 --write-logs
    python -m nuru simulate --power --groups 8 --p-respond 0.5,0.5,0.7
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .design import schedule
from .eventlog import FIELDS
from .controller import auto_invite_rule
from .participation import Tracker, gini

DEFAULT_MOD = {"auto_invite_silence_s": 7.0, "turn_end_gap_s": 1.5,
               "turn_end_invite_interval_s": 40.0, "min_gap_between_invites_s": 12.0}
MEASURES = ["gini_time", "gini_turns", "least_active_share_turns", "response_rate"]

PERSONAS = {  # voluntary start prob per s (when silent), mean turn length (s), respond multiplier
    "dominant": (0.12, 14.0, 1.0),
    "average": (0.05, 8.0, 0.9),
    "quiet": (0.012, 5.0, 0.6),
}
GROUP = {"A": "average", "B": "dominant", "C": "quiet", "D": "average"}
POLICY = {"c1": "round_robin", "c2": "least_active", "c3": "least_active"}


class SimClock:
    def __init__(self, t0=1_700_000_000.0):
        self.t = t0

    def __call__(self):
        return self.t


def simulate_block(condition: str, p_respond: float, rng: random.Random, duration_s=600,
                   mod: dict | None = None, rows=None, group="SIM", task=""):
    mod = mod or DEFAULT_MOD
    clock = SimClock(rng.uniform(1.7e9, 1.8e9))
    seats = list(GROUP)
    tr = Tracker(seats, clock=clock)
    t_end = clock.t + duration_s
    last_invite = -1e9
    turn_end = None

    def w(event, seat="", value=""):
        if rows is not None:
            rows.append({"t": f"{clock.t:.3f}", "t_rel": "", "group": group, "condition": condition,
                         "task": task, "event": event, "seat": seat, "value": value})

    w("condition_start", value=json.dumps({"task": task}))
    while clock.t < t_end:
        if tr.current:
            if clock.t >= turn_end:
                w("speak_stop", tr.current)
                tr.stop()
        else:
            starters = [s for s in seats if rng.random() < PERSONAS[GROUP[s]][0]]
            rule = auto_invite_rule(mod, False, tr.silence_s(), clock.t - last_invite)
            if rule:   # the robot takes the floor first when a rule fires
                starters = []
            if starters:
                s = rng.choice(starters)
                info = tr.start(s)
                w("speak_start", s)
                if info.get("response"):
                    w("response", s, f"{info['latency_s']:.3f}")
                turn_end = clock.t + rng.expovariate(1 / PERSONAS[GROUP[s]][1])
            elif rule:
                seat = tr.choose(POLICY[condition])
                tr.register_invite(seat)
                w("invite", seat, f"auto:{rule}")
                last_invite = clock.t
                clock.t += 4.0                       # robot prompt duration
                tr.open_response_window(seat)
                if rng.random() < p_respond * PERSONAS[GROUP[seat]][2]:
                    clock.t += rng.uniform(0.5, 3.0)
                    info = tr.start(seat)
                    w("speak_start", seat)
                    if info.get("response"):
                        w("response", seat, f"{info['latency_s']:.3f}")
                    turn_end = clock.t + rng.expovariate(1 / max(4.0, PERSONAS[GROUP[seat]][1] * 0.8))
                continue
        clock.t += 1.0
    if tr.current:
        w("speak_stop", tr.current)
    tr.stop()
    w("metrics", value=json.dumps(tr.metrics()))
    w("condition_end")
    return tr.metrics()


def run(n_groups=10, p_respond=(0.5, 0.5, 0.5), seed=1, write_logs: Path | None = None, verbose=True,
        mod: dict | None = None):
    rng = random.Random(seed)
    pr = dict(zip(("c1", "c2", "c3"), p_respond))
    results = []
    for g in schedule(n_groups):
        gid = g["group"].replace("G", "SIM")
        rows = [] if write_logs else None
        if rows is not None:
            rows.append({"t": "1700000000.000", "t_rel": "", "group": gid, "condition": "", "task": "",
                         "event": "session_start", "seat": "", "value": json.dumps({"seats": list(GROUP)})})
            for s, p in GROUP.items():
                rows.append({"t": "1700000000.000", "t_rel": "", "group": gid, "condition": "", "task": "",
                             "event": "participant_meta", "seat": s,
                             "value": {"dominant": "active"}.get(p, p)})
        for b in g["blocks"]:
            m = simulate_block(b["condition"], pr[b["condition"]], rng, mod=mod, rows=rows, group=gid, task=b["task"])
            results.append({"group": gid, "condition": b["condition"],
                            **{k: m[k] for k in MEASURES}, "invitations": sum(m["invitations"].values())})
        if write_logs:
            write_logs.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda r: float(r["t"]))
            with open(write_logs / f"nuru_{gid}_sim.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=FIELDS)
                wr.writeheader()
                wr.writerows(rows)
    if verbose:
        cols = MEASURES + ["invitations"]
        print("      " + "".join(f"{k:>26}" for k in cols))
        for c in ("c1", "c2", "c3"):
            rs = [r for r in results if r["condition"] == c]
            cells = []
            for k in cols:
                v = [r[k] for r in rs if r[k] is not None]
                cells.append(f"{sum(v) / len(v):26.3f}" if v else " " * 25 + "-")
            print(f"  {c.upper()}" + "".join(cells) + f"   (p_respond={pr[c]})")
    return results


def power(n_groups=8, p_respond=(0.5, 0.5, 0.7), reps=300, alpha=0.05, seed=7, mod=None):
    """Share of simulated studies in which each planned contrast is significant."""
    from scipy import stats
    hits = {(m, c): 0 for m in MEASURES for c in ("c1_vs_c2", "c2_vs_c3")}
    rng = random.Random(seed)
    for _ in range(reps):
        res = run(n_groups, p_respond, seed=rng.randrange(1 << 30), verbose=False, mod=mod)
        by = {}
        for r in res:
            by.setdefault(r["group"], {})[r["condition"]] = r
        for m in MEASURES:
            for name, a, b in (("c1_vs_c2", "c1", "c2"), ("c2_vs_c3", "c2", "c3")):
                pairs = [(d[a][m], d[b][m]) for d in by.values() if d[a][m] is not None and d[b][m] is not None]
                try:
                    if len(pairs) >= 5 and stats.wilcoxon(*zip(*pairs)).pvalue < alpha:
                        hits[(m, name)] += 1
                except ValueError:
                    pass
    print(f"Power with {n_groups} groups, p_respond(C1,C2,C3)={p_respond}, {reps} simulated studies, alpha={alpha}")
    print(f"  {'measure':<26}{'C1 vs C2':>10}{'C2 vs C3':>10}")
    for m in MEASURES:
        print(f"  {m:<26}{hits[(m, 'c1_vs_c2')] / reps:>10.2f}{hits[(m, 'c2_vs_c3')] / reps:>10.2f}")
    return {f"{m}:{c}": v / reps for (m, c), v in hits.items()}
