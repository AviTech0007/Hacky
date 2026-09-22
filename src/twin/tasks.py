"""The unit of work the coordinator schedules.

A Task is deliberately simple: constant power draw for a whole number of
hourly slots, inside a release/deadline window. That keeps the MILP linear
while still capturing every trade-off we care about.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

PRIORITY_VALUE = {
    "critical": 20.0,   # safety/telemetry - effectively a hard requirement
    "high": 6.0,
    "flexible": 1.0,    # nice to have, drop it when energy is scarce
}


@dataclass
class Task:
    id: str
    name: str
    kind: str            # sensing | inference | upload | maintenance
    priority: str        # critical | high | flexible
    power_kw: float
    duration_h: int
    release_h: int       # earliest slot it may start
    deadline_h: int      # last slot it may still be running in
    preemptible: bool = True

    @property
    def energy_kwh(self) -> float:
        return self.power_kw * self.duration_h

    @property
    def value(self) -> float:
        return PRIORITY_VALUE[self.priority]

    @property
    def slack_h(self) -> int:
        """How much room the scheduler has to move this task around."""
        return (self.deadline_h - self.release_h + 1) - self.duration_h

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["energy_kwh"] = round(self.energy_kwh, 3)
        d["slack_h"] = self.slack_h
        return d


def default_workload(horizon_h: int = 24, seed: int = 7, n_flexible: int = 6) -> List[Task]:
    """A realistic off-grid edge node's day.

    Fixed backbone tasks (telemetry, nightly upload) plus a random tail of
    deferrable compute. The fixed ones make the demo reproducible; the random
    ones prove the optimizer is doing real work.
    """
    import random

    rng = random.Random(seed)
    tasks: List[Task] = [
        Task("T01", "Safety telemetry beacon", "sensing", "critical",
             power_kw=0.10, duration_h=2, release_h=0, deadline_h=5, preemptible=True),
        Task("T02", "Structural vibration sweep", "sensing", "critical",
             power_kw=0.25, duration_h=3, release_h=2, deadline_h=11, preemptible=False),
        Task("T03", "Satellite uplink window", "upload", "critical",
             power_kw=0.45, duration_h=1, release_h=19, deadline_h=21, preemptible=False),
        Task("T04", "Vision model inference batch", "inference", "high",
             power_kw=0.70, duration_h=3, release_h=0, deadline_h=17, preemptible=True),
        Task("T05", "Firmware integrity check", "maintenance", "high",
             power_kw=0.30, duration_h=2, release_h=6, deadline_h=20, preemptible=True),
    ]

    kinds = ["inference", "upload", "maintenance", "sensing"]
    for i in range(n_flexible):
        duration = rng.choice([1, 1, 2, 3])
        release = rng.randint(0, max(0, horizon_h - duration - 6))
        deadline = min(horizon_h - 1, release + duration + rng.randint(3, 10))
        tasks.append(
            Task(
                id=f"F{i + 1:02d}",
                name=f"{rng.choice(kinds).title()} job {i + 1}",
                kind=rng.choice(kinds),
                priority="flexible",
                power_kw=round(rng.uniform(0.15, 0.80), 2),
                duration_h=duration,
                release_h=release,
                deadline_h=deadline,
                preemptible=True,
            )
        )
    return tasks


def shift_tasks(tasks: List[Task], by_hours: int, horizon_h: int) -> List[Task]:
    """Re-base a task list onto a new time origin (used by the rolling loop)."""
    out: List[Task] = []
    for t in tasks:
        release = max(0, t.release_h - by_hours)
        deadline = t.deadline_h - by_hours
        if deadline < 0:
            continue  # window has passed
        out.append(
            Task(t.id, t.name, t.kind, t.priority, t.power_kw, t.duration_h,
                 release, min(deadline, horizon_h - 1), t.preemptible)
        )
    return out
