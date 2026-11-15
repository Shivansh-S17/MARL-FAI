"""Static research figures that do not depend on results: labelled map + information diagram."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from ..env.geometry import SIZE, cell, load_map, xy

NPC_COL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
PLAYER = "#D55E00"
WP = [(3, 3), (10, 3), (16, 3), (16, 16), (10, 16), (3, 16)]


def _base(ax, gm):
    img = np.where(gm.walls, 0.45, 0.98)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="upper", extent=(-0.5, SIZE - 0.5, SIZE - 0.5, -0.5))
    ax.set_xticks(range(0, SIZE, 2)); ax.set_yticks(range(0, SIZE, 2))
    ax.tick_params(labelsize=7)
    ax.grid(color="#dddddd", lw=0.4)
    ax.set_xlim(-0.5, SIZE - 0.5); ax.set_ylim(SIZE - 0.5, -0.5)


def render_map_figures(out: Path):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    gm = load_map("corridor")
    vis = gm.visibility(5)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6))
    # (a) labelled map
    ax = axs[0]; _base(ax, gm)
    for d in gm.doorways:
        x, y = xy(d); ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fc="#9ad0f3", ec="#0072B2", lw=1))
    ax.plot([p[0] for p in WP + [WP[0]]], [p[1] for p in WP + [WP[0]]], "--", color=PLAYER, lw=1.5, label="scripted player route")
    for k, (x, y) in enumerate(WP):
        ax.plot(x, y, "o", color=PLAYER, ms=4); ax.text(x + .3, y - .4, f"w{k + 1}", fontsize=7, color=PLAYER)
    ax.set_title("(a) Corridor map: walls, 6 doorways (blue), player route", fontsize=9)
    ax.plot([], [], "s", color="#9ad0f3", mec="#0072B2", label="doorway")
    ax.legend(loc="lower center", fontsize=7, ncol=2, frameon=True)
    # (b) information diagram
    ax = axs[1]; _base(ax, gm)
    npcs = [(2, 6), (9, 4), (15, 9), (9, 16)]
    player = (10, 7)
    for i, (x, y) in enumerate(npcs):
        v = vis[cell(x, y)].reshape(SIZE, SIZE)
        ov = np.zeros((SIZE, SIZE, 4)); ov[v] = matplotlib.colors.to_rgba(NPC_COL[i], 0.17)
        ax.imshow(ov, origin="upper", extent=(-0.5, SIZE - 0.5, SIZE - 0.5, -0.5), zorder=2)
    marks = ["o", "s", "^", "D"]
    for i, (x, y) in enumerate(npcs):
        ax.plot(x, y, marks[i], color=NPC_COL[i], mec="black", ms=11, zorder=5)
        ax.text(x, y, str(i + 1), color="white", fontsize=7, ha="center", va="center", zorder=6, weight="bold")
    ax.plot(*player, "*", color=PLAYER, mec="black", ms=17, zorder=7)
    seen_by = [i for i, (x, y) in enumerate(npcs) if vis[cell(x, y), cell(*player)]]
    for i in seen_by:
        ax.add_patch(FancyArrowPatch(npcs[i], player, arrowstyle="->", color=NPC_COL[i], lw=1.2, mutation_scale=10, zorder=4))
    others = [i for i in range(4) if i not in seen_by]
    for i in others[:2]:
        for s in seen_by[:1]:
            ax.add_patch(FancyArrowPatch(npcs[s], npcs[i], arrowstyle="->", color=NPC_COL[s], lw=1.2, ls="--",
                                         mutation_scale=10, connectionstyle="arc3,rad=0.15", zorder=4))
    ax.set_title("(b) Information: shaded = each NPC's view (radius 5, line of sight).\n"
                 "Solid arrow: direct sighting. Dashed arrow: budgeted 16-byte packet.", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"map_information.{ext}", dpi=170)
    plt.close(fig)
