"""Study design helpers: counterbalanced condition order and topic assignment.

Within-subjects, 3 conditions. A single 3x3 Latin square balances *position*
but not *carry-over* (which condition comes right before which). A Williams
design does both; for an odd number of conditions it needs two squares, i.e.
6 orders. Groups cycle through these 6 orders.

Topics: the topic order is fixed by position (1st discussion = T1, ...), so
across every 6 groups each condition is paired with each topic equally often
(2x). Condition and topic are therefore not confounded.
"""

from __future__ import annotations

CONDITIONS = ["c1", "c2", "c3"]
TOPICS = ["T1", "T2", "T3"]


def williams_orders(items: list[str]) -> list[list[str]]:
    n = len(items)
    # standard Williams first row: 0, 1, n-1, 2, n-2, ...
    first, lo, hi = [0], 1, n - 1
    while len(first) < n:
        first.append(lo); lo += 1
        if len(first) < n:
            first.append(hi); hi -= 1
    rows = [[(x + r) % n for x in first] for r in range(n)]
    if n % 2 == 1:
        rows += [list(reversed(r)) for r in rows]
    return [[items[i] for i in r] for r in rows]


def schedule(n_groups: int, conditions=None, topics=None) -> list[dict]:
    conditions = conditions or CONDITIONS
    topics = topics or TOPICS
    orders = williams_orders(conditions)
    plan = []
    for g in range(n_groups):
        order = orders[g % len(orders)]
        plan.append({
            "group": f"G{g + 1:02d}",
            "blocks": [{"position": i + 1, "condition": c, "task": topics[i]} for i, c in enumerate(order)],
        })
    return plan


def print_schedule(n_groups: int):
    plan = schedule(n_groups)
    print("Group | Block 1     | Block 2     | Block 3")
    print("------+-------------+-------------+------------")
    for p in plan:
        cells = [f"{b['condition'].upper()} + {b['task']}" for b in p["blocks"]]
        print(f"{p['group']}   | " + " | ".join(f"{c:<11}" for c in cells))
    return plan
