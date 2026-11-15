"""Grid geometry: walls, graph distances, supercover line of sight, visibility."""
from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from ..config import MAP_DIR

SIZE = 20
NC = SIZE * SIZE
# action order: stay, north, east, south, west ; (dx, dy), y grows downward
ACTIONS = ((0, 0), (0, -1), (1, 0), (0, 1), (-1, 0))
ACTION_NAMES = ("stay", "north", "east", "south", "west")


def cell(x: int, y: int) -> int:
    return y * SIZE + x


def xy(c: int) -> tuple[int, int]:
    return c % SIZE, c // SIZE


def supercover(x0, y0, x1, y1):
    """All grid cells touched by the segment between cell centres (corner touches
    include both cells adjacent to the corner)."""
    dx, dy = x1 - x0, y1 - y0
    nx, ny = abs(dx), abs(dy)
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    x, y = x0, y0
    out = [(x, y)]
    ix = iy = 0
    while ix < nx or iy < ny:
        d = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if d == 0:  # passes exactly through a corner
            out.append((x + sx, y))
            out.append((x, y + sy))
            x += sx
            y += sy
            ix += 1
            iy += 1
        elif d < 0:
            x += sx
            ix += 1
        else:
            y += sy
            iy += 1
        out.append((x, y))
    return out


class GridMap:
    def __init__(self, walls: np.ndarray, name: str = "custom"):
        assert walls.shape == (SIZE, SIZE)
        self.name = name
        self.walls = walls.astype(bool)  # [y, x]
        self.wall_flat = self.walls.reshape(-1)
        self.free_cells = np.flatnonzero(~self.wall_flat)
        self.free_index = -np.ones(NC, dtype=np.int32)
        self.free_index[self.free_cells] = np.arange(len(self.free_cells))
        self.n_free = len(self.free_cells)
        # nxt[c, a] -> resulting cell, -1 if illegal
        self.nxt = -np.ones((NC, 5), dtype=np.int32)
        for c in self.free_cells:
            x, y = xy(int(c))
            for a, (dx, dy) in enumerate(ACTIONS):
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < SIZE and 0 <= ny_ < SIZE and not self.walls[ny_, nx_]:
                    self.nxt[c, a] = cell(nx_, ny_)
        self.legal = self.nxt >= 0
        self._bfs_all()
        self.doorways = self._find_doorways()
        self._vis_cache: dict[int, np.ndarray] = {}
        self._build_helpers()

    # ------------------------------------------------------------------ build
    def _bfs_all(self):
        dist = -np.ones((NC, NC), dtype=np.int16)
        for s in self.free_cells:
            d = dist[s]
            d[s] = 0
            q = deque([int(s)])
            while q:
                c = q.popleft()
                for a in range(1, 5):
                    n = self.nxt[c, a]
                    if n >= 0 and d[n] < 0:
                        d[n] = d[c] + 1
                        q.append(int(n))
        self.dist = dist
        sub = dist[np.ix_(self.free_cells, self.free_cells)]
        self.connected = bool((sub >= 0).all())
        self.diameter = int(sub.max())

    def _find_doorways(self):
        doors = []
        for c in self.free_cells:
            x, y = xy(int(c))
            if 0 < y < SIZE - 1 and self.walls[y - 1, x] and self.walls[y + 1, x]:
                doors.append(int(c))
            elif 0 < x < SIZE - 1 and self.walls[y, x - 1] and self.walls[y, x + 1]:
                doors.append(int(c))
        return sorted(doors)

    def _build_helpers(self):
        # best shortest-path action table with fixed-order tie breaking
        big = 10_000
        d = np.where(self.dist < 0, big, self.dist).astype(np.int32)
        cand = np.full((5, NC, NC), big * 2, dtype=np.int32)
        for a in range(5):
            ok = self.nxt[:, a] >= 0
            cand[a, ok, :] = d[self.nxt[ok, a], :]
        self.best_action = np.argmin(cand, axis=0).astype(np.int8)  # [cell, dest]
        # nearest doorway to every cell (ties: lowest cell index)
        if self.doorways:
            dd = self.dist[:, self.doorways].astype(np.int32)
            dd[dd < 0] = big
            self.nearest_door = np.array(self.doorways)[np.argmin(dd, axis=1)]
        else:
            self.nearest_door = -np.ones(NC, dtype=np.int32)
        # quadrant cell lists (split at x=10, y=10), index = (y>=10)*2+(x>=10)
        self.quadrants = []
        for q in range(4):
            qs = [int(c) for c in self.free_cells if ((xy(int(c))[1] >= 10) * 2 + (xy(int(c))[0] >= 10)) == q]
            self.quadrants.append(np.array(qs, dtype=np.int32))
        # coarse 4x4 region index
        self.region = np.array([(xy(c)[1] // 5) * 4 + xy(c)[0] // 5 for c in range(NC)], dtype=np.int32)
        self.xs = np.array([c % SIZE for c in range(NC)], dtype=np.int32)
        self.ys = np.array([c // SIZE for c in range(NC)], dtype=np.int32)
        # compact transition table in free-index space (for belief propagation)
        fi = self.free_index
        nf = np.where(self.nxt[self.free_cells] >= 0, fi[np.maximum(self.nxt[self.free_cells], 0)], -1)
        self.nxt_free = nf.astype(np.int32)  # [F, 5]
        self.legal_free = self.legal[self.free_cells]  # [F, 5]

    # ------------------------------------------------------------- visibility
    def line_clear(self, a: int, b: int) -> bool:
        x0, y0 = xy(a)
        x1, y1 = xy(b)
        for (cx, cy) in supercover(x0, y0, x1, y1)[1:-1]:
            if self.walls[cy, cx]:
                return False
        return True

    def visibility(self, radius: int) -> np.ndarray:
        """vis[a, b]: b is visible from a (Manhattan <= r and clear supercover line)."""
        if radius in self._vis_cache:
            return self._vis_cache[radius]
        vis = np.zeros((NC, NC), dtype=bool)
        fc = [int(c) for c in self.free_cells]
        for a in fc:
            ax, ay = xy(a)
            for b in fc:
                bx, by = xy(b)
                if abs(ax - bx) + abs(ay - by) <= radius and self.line_clear(a, b):
                    vis[a, b] = True
        self._vis_cache[radius] = vis
        return vis

    def legal_actions(self, c: int):
        return np.flatnonzero(self.legal[c])


@lru_cache(maxsize=None)
def load_map(name: str) -> GridMap:
    rows = (MAP_DIR / f"{name}.txt").read_text().strip().splitlines()
    assert len(rows) == SIZE and all(len(r) == SIZE for r in rows), "map must be 20x20"
    walls = np.array([[ch == "#" for ch in r] for r in rows])
    return GridMap(walls, name)
