import numpy as np

from pursuit.config import load_config
from pursuit.env.geometry import cell, load_map
from pursuit.env.scenario import room_of, sample_for_cfg
from pursuit.env.team import make_core


def test_split_sampler_constraints():
    cfg = load_config(overrides=dict(env=dict(radius=3, horizon=80, spawn="split")))
    gm = load_map("corridor")
    vis = gm.visibility(3)
    rng = np.random.default_rng(0)
    for _ in range(300):
        s = sample_for_cfg(cfg, gm, rng)
        o = s.meta["observer"]
        assert gm.dist[s.npc_cells[o], s.player_cell] == 3 and vis[s.npc_cells[o], s.player_cell]
        others = [c for i, c in enumerate(s.npc_cells) if i != o]
        for c in others:
            assert 8 <= gm.dist[c, s.player_cell] <= 12 and not vis[c, s.player_cell]
        assert len(set(s.npc_cells)) == 4 and s.player_cell not in s.npc_cells
        assert len({room_of(c) for c in others} - {-1}) >= 2


def test_player_visibility_independent_of_npc_radius():
    # player at distance 5 in the open corridor row: NPC radius 3 cannot see it, the player (radius 5) sees the NPC
    cfg = load_config(overrides=dict(env=dict(radius=3)))
    core = make_core(cfg)
    a, b = cell(1, 8), cell(5, 9)          # same room, Manhattan distance 5
    assert not core.vis[a, b] and core.pvis[a, b]
    core.npc, core.pcell, core.t = [b, cell(1, 1), cell(2, 1), cell(3, 1)], a, 0
    assert (0, b) in core.player_obs().npcs   # the player still sees at radius 5
    assert core.observe(0).player_cell == -1  # the NPC (radius 3) does not see the player


def test_bootstrap_handles_undefined_gap_fractions():
    # every method captures everything at every budget: gap denominators are zero, Fgap must stay undefined
    from pursuit.evaluation.analysis import bootstrap
    C = np.ones((4, 3, 10))
    res = bootstrap(C, C, nboot=50)
    assert np.isnan(res["F"]["point"]).all() and np.isnan(res["F"]["lo"]).all()
    assert res["F"]["undefined_frac"].min() == 1.0
    assert np.allclose(res["G"]["point"], 0)
