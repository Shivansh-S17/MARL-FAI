import numpy as np
import pytest

from pursuit.config import load_config
from pursuit.env.geometry import SIZE, GridMap, cell, load_map
from pursuit.env.scenario import Scenario
from pursuit.env.team import Team, make_core


@pytest.fixture(scope="session")
def gm():
    return load_map("corridor")


def mk(budget="1", **env):
    cfg = load_config(overrides=dict(env=dict(budget=budget, **env)))
    core = make_core(cfg)
    return cfg, core, Team(core)


def open_map(walls=()):
    w = np.zeros((SIZE, SIZE), bool)
    w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = True
    for x, y in walls:
        w[y, x] = True
    return GridMap(w, "test")


def scen(npcs, player, phase=0, seed=1):
    return Scenario([cell(*p) for p in npcs], cell(*player), phase, seed, "t")
