"""Comparison scenes built from actual recorded evaluation traces (Section 9.3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import ROOT, load_config, parse_budget
from ..env.geometry import load_map
from ..opponents.players import WaypointPlayer
from ..rendering.trace import TraceRecorder
from .play import RuleController, make_scenarios, play, build_team
from ..env.team import make_core

REPLAYS = ROOT / "evaluator_pack" / "replays"


def record_rule(cfg, scenario, rule, B, player, path, meta_extra=None):
    core = make_core(cfg, player=player, budget=B)
    team = build_team(cfg, core, None)
    rec = TraceRecorder()
    r = play(core, team, scenario, RuleController(rule), rec)
    meta = dict(map=cfg.env.map, B=str(parse_budget(B)), method=rule, run_id=rule, scenario=scenario.to_dict(),
                protocol="frozen (rule-based)", episode=1, **(meta_extra or {}))
    rec.finalize(path, meta)
    return r


def first_local_only_scenario(cfg, B="1/2", n=200, seed0=100000):
    """First scenario where, at some step, the player is visible to exactly one NPC."""
    gm = load_map(cfg.env.map)
    for sc in make_scenarios(cfg, seed0, n, "f"):
        core = make_core(cfg, player=WaypointPlayer(gm, 0.25), budget=B)
        team = build_team(cfg, core, None)
        rec = TraceRecorder()
        play(core, team, sc, RuleController("RC"), rec)
        if any(sum(1 for d in row["direct"] if d >= 0) == 1 for row in rec.rows):
            return sc
    return None


def build_scene1(cfg=None, out=REPLAYS):
    """Local-information scene: deterministic illustration with RC (clearly labelled)."""
    cfg = cfg or load_config()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sc = first_local_only_scenario(cfg)
    gm = load_map(cfg.env.map)
    p = WaypointPlayer(gm, 0.25)
    r = record_rule(cfg, sc, "RC", "1/2", p, out / "scene1_local_info.npz",
                    dict(player_label="Scripted waypoint player (xi = 0.25)",
                         note="Press V then 1-4: what each NPC knows", caption="Illustrative: local information"))
    (out / "scene1_local_info.json").write_text(json.dumps(dict(
        title="Local information", kind="illustrative", traces=["scene1_local_info.npz"],
        caption="Illustrative deterministic example (RC, B=1/2)",
        card=["Each NPC sees only radius 5 with line of sight.",
              "Press V and 1-4 in the viewer to see one NPC's view.",
              "Dashed arrows: packets sent. Solid arrows: packets delivered."])))
    return r
