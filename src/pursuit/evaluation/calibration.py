"""Message-use audit and bounded task calibration (plan Section 6.1). Validation seeds only (>= 50000)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, load_config
from ..controllers.baselines import RCController
from ..env.geometry import cell, load_map
from ..env.scenario import Scenario
from ..env.team import Team, make_core
from ..opponents.players import WaypointPlayer
from .play import RuleController, make_scenarios

CALIB_SEED = 50000
CANDIDATES = {   # predeclared, in order (plan 6.1); corridor map, N=4, delta=0, equal speeds
    "C0": dict(radius=5, horizon=200, spawn="standard"),
    "C1": dict(radius=3, horizon=80, spawn="standard"),
    "C2": dict(radius=3, horizon=80, spawn="split"),
    "C3": dict(radius=3, horizon=120, spawn="split"),
}


def run_episode_audited(core, team, sc, ctrl):
    """Rule-controller episode that also records sighting and delivery statistics."""
    core.reset(sc)
    team.reset()
    ctrl.begin_episode()
    first_sight, sight_steps, received = -1, 0, 0
    while not core.done:
        team.begin_step()
        any_vis = any(inf.player_visible for inf in team.infos)
        if any_vis:
            sight_steps += 1
            if first_sight < 0:
                first_sight = core.t
        received += sum(len(inf.last_delivered_raw) for inf in team.infos)
        moves, sends = ctrl.act_team(team)
        team.commit_and_step(moves, sends)
    return dict(captured=int(core.captured), capped_duration=core.t, packets=core.channel.total_sent,
                received=received, first_sighting=first_sight if first_sight >= 0 else core.t,
                sighting_fraction=sight_steps / max(core.t, 1))


def block(cfg, scen, xi, rule="RC"):
    core = make_core(cfg, player=WaypointPlayer(load_map(cfg.env.map), xi))
    team = Team(core)
    ctrl = RuleController(rule)
    return pd.DataFrame([run_episode_audited(core, team, s, ctrl) for s in scen])


def deterministic_counterexample(B):
    """NPC 0 sees the player; NPC 1 (same room, out of sight) can only learn of it through a packet."""
    cfg = load_config(overrides=dict(env=dict(budget=B, radius=5)))
    core = make_core(cfg, player=WaypointPlayer(load_map("corridor"), 0.0))
    team = Team(core)
    sc = Scenario([cell(2, 2), cell(3, 8), cell(16, 15), cell(10, 12)], cell(5, 2), 0, 1, "ce")
    ctrl = RuleController("RC")
    core.reset(sc)
    team.reset()
    out = dict(B=B)
    team.begin_step()
    out["npc0_sees"] = bool(team.infos[0].player_visible)
    out["npc1_mem_before"] = int(team.infos[1].mem_cell)
    out["npc1_dest_before"] = int(team.infos[1].dest)
    moves, sends = ctrl.act_team(team)
    out["sends_t0"] = [bool(s) for s in sends]
    team.commit_and_step(moves, sends)
    team.begin_step()
    i1 = team.infos[1]
    out["npc1_mem_after"] = int(i1.mem_cell)
    out["npc1_mem_step"] = int(i1.mem_step)
    out["npc1_received"] = len(i1.last_delivered_raw)
    out["npc1_dest_after"] = int(i1.dest)
    out["npc1_role_after"] = int(i1.role)
    return out


def audit():
    res = dict(counterexample_B0=deterministic_counterexample("0"), counterexample_B1=deterministic_counterexample("1"))
    c0, c1 = res["counterexample_B0"], res["counterexample_B1"]
    res["pass_B0_silent"] = (not any(c0["sends_t0"])) and c0["npc1_received"] == 0 and c0["npc1_mem_after"] == -1
    res["pass_B1_delivers"] = bool(c1["sends_t0"][0] and c1["npc1_received"] >= 1
                                   and c1["npc1_mem_after"] == cell(5, 2) and c1["npc1_mem_step"] == 0)
    res["pass_role_hint_changes"] = bool(c1["npc1_dest_after"] != c0["npc1_dest_after"]
                                         and c1["npc1_role_after"] != c0["npc1_role_after"])
    return res


def paired_bootstrap(g0, g1, reps=5000, seed=40000):
    rng = np.random.default_rng(seed)
    d = np.asarray(g1, float) - np.asarray(g0, float)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def calibrate(n=300, out=None, candidates=CANDIDATES):
    rows = []
    for cid, c in candidates.items():
        for xi in (0.0, 0.75):
            res = {}
            for B in ("0", "1"):
                cfg = load_config(overrides=dict(env=dict(budget=B, **c)))
                scen = make_scenarios(cfg, CALIB_SEED, n, "cal")
                res[B] = block(cfg, scen, xi)
            g, lo, hi = paired_bootstrap(res["0"].captured, res["1"].captured)
            cap1 = float(res["1"].captured.mean())
            promising = bool(0.20 <= cap1 <= 0.95 and g >= 0.10 and lo > 0)
            rows.append(dict(cand=cid, xi=xi, **c, cap_B0=float(res["0"].captured.mean()), cap_B1=cap1, gain=g,
                             gain_lo=lo, gain_hi=hi, dur_B0=float(res["0"].capped_duration.mean()),
                             dur_B1=float(res["1"].capped_duration.mean()),
                             first_sight_B1=float(res["1"].first_sighting.mean()),
                             sight_frac_B1=float(res["1"].sighting_fraction.mean()),
                             packets_B0=float(res["0"].packets.mean()), packets_B1=float(res["1"].packets.mean()),
                             received_B0=float(res["0"].received.mean()), received_B1=float(res["1"].received.mean()),
                             promising=promising))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    a = audit()
    print(json.dumps(a, indent=1))
    if len(sys.argv) > 1:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        df = calibrate(n, sys.argv[1])
        ok = df[df.groupby("cand").promising.transform("any")]
        print(df.round(3).to_string())
