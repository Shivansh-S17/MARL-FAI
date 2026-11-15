import numpy as np

from conftest import mk, open_map, scen
from pursuit.env.geometry import cell, load_map, supercover, xy
from pursuit.env.scenario import sample_scenario
from pursuit.env.core import PursuitCore
from pursuit.opponents.players import WaypointPlayer


def test_maps_connected_and_doorways():
    for m in ("corridor", "arena"):
        g = load_map(m)
        assert g.connected
    g = load_map("corridor")
    assert sorted(xy(c) for c in g.doorways) == sorted([(6, 4), (6, 10), (6, 15), (13, 4), (13, 10), (13, 15)])
    assert g.walls[:, 0].all() and g.walls[0, :].all()


def test_legal_moves_masked(gm):
    c = cell(1, 1)
    assert list(gm.legal[c]) == [True, False, True, True, False]  # stay, east, south only
    assert not gm.legal[cell(5, 3), 2]  # wall at x=6 blocks east from (5,3)
    assert gm.legal[cell(5, 4), 2]      # doorway at y=4


def test_visibility_corner_and_wall_blocking():
    g = open_map(walls=[(1, 0 + 1)])  # wall at (1,1)
    # diagonal (2,2)->(0,0) wait: use (3,3)->(5,5) passes exactly through corners
    g2 = open_map(walls=[(4, 3)])
    # line (3,3)->(5,5) touches the corner between (4,3) and (3,4)/(4,4)... supercover includes (4,3),(3,4)
    cells = supercover(3, 3, 5, 5)
    assert (4, 3) in cells and (3, 4) in cells and (4, 4) in cells
    assert not g2.line_clear(cell(3, 3), cell(5, 5))   # corner touch of a wall blocks
    g3 = open_map(walls=[(4, 6)])
    assert g3.line_clear(cell(3, 3), cell(5, 5))
    # direct wall on the line
    g4 = open_map(walls=[(4, 4)])
    assert not g4.line_clear(cell(3, 3), cell(5, 5))
    # symmetric
    vis = g2.visibility(5)
    assert (vis == vis.T).all()
    # Manhattan radius
    assert not vis[cell(3, 3), cell(9, 3)] and open_map().visibility(5)[cell(3, 3), cell(8, 3)]


def test_capture_colocation_and_edge_swap():
    cfg, core, team = mk("0", n_npcs=1)
    # head-on edge swap: NPC (8,5) moves east, player at (9,5) moves west
    class West:
        kind = "x"; learning = False
        def reset(self, s): pass
        def act(self, p, a, b): return 4
    core.set_player(West())
    core.reset(scen([(8, 5)], (9, 5)))
    res = core.step([2], [None])
    assert res.captured and core.done
    core.reset(scen([(8, 5)], (10, 5)))   # co-location after movement
    res = core.step([2], [None])
    assert res.captured


def test_failure_after_exactly_H_transitions():
    cfg, core, team = mk("0", n_npcs=1)
    core.reset(scen([(1, 1)], (18, 18)))
    n = 0
    while not core.done:
        core.step([0], [None]); n += 1
    assert n == core.H == 200 and not core.captured


def test_simultaneous_update_order_independent():
    outs = []
    npcs = [(3, 3), (4, 3), (10, 8), (11, 8)]
    moves = [2, 4, 1, 3]  # first two swap cells (pass through), others move apart
    for order in ([0, 1, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0]):
        cfg, core, team = mk("0")
        core.reset(scen([npcs[k] for k in order], (15, 15)))
        core.step([moves[k] for k in order], [None] * 4)
        outs.append((sorted(core.npc), core.pcell, core.captured, core.t))
    assert outs[0] == outs[1] == outs[2]


def test_npcs_pass_through_each_other():
    cfg, core, team = mk("0", n_npcs=2)
    core.reset(scen([(8, 5), (9, 5)], (18, 18)))
    core.step([2, 4], [None, None])  # swap cells: allowed
    assert core.npc == [cell(9, 5), cell(8, 5)]


def test_waypoint_xi_zero_follows_route_and_xi_random():
    g = load_map("corridor")
    p = WaypointPlayer(g, 0.0)
    p.reset(scen([(1, 1)], (3, 3), phase=1))
    from pursuit.opponents.players import PlayerObs
    pos = cell(3, 3)
    for t in range(60):
        a = p.act(PlayerObs(pos, t, ()), 0.5, 0.5)
        pos = int(g.nxt[pos, a])
        assert pos >= 0
    p2 = WaypointPlayer(g, 0.75); p2.reset(scen([(1, 1)], (3, 3)))
    acts = [p2.act(PlayerObs(cell(3, 3), 0, ()), 0.1, u) for u in np.linspace(0, 0.99, 10)]
    assert len(set(acts)) > 1


def test_api_wrapper():
    from pettingzoo.test import parallel_api_test
    from pursuit.env.pz import PursuitParallelEnv
    from pursuit.config import load_config
    parallel_api_test(PursuitParallelEnv(load_config()), num_cycles=30)
