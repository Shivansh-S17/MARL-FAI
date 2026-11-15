"""One reusable renderer: pygame display, headless PNG keyframes and GIF export all use this."""
from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame

from ..env.geometry import SIZE, load_map, xy
from .trace import Trace

W, H = 960, 540
# Okabe-Ito colour-blind safe palette; identity is also encoded by shape + printed ID.
NPC_COL = [(0, 114, 178), (230, 159, 0), (0, 158, 115), (204, 121, 167)]
NPC_SHAPE = ["circle", "square", "triangle", "diamond"]
PLAYER_COL = (213, 94, 0)
BG = (250, 250, 248)
WALL = (110, 110, 115)
GRID = (226, 226, 222)
INK = (30, 30, 34)
MUTE = (110, 110, 118)
ROLE_NAMES = ["chase", "intercept", "block", "search"]
ACTION_NAMES = ["stay", "N", "E", "S", "W"]


def init_pygame(headless: bool):
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.font.init()


class Fonts:
    def __init__(self):
        self.s = pygame.font.SysFont("arial", 13)
        self.m = pygame.font.SysFont("arial", 16)
        self.b = pygame.font.SysFont("arial", 18, bold=True)
        self.xs = pygame.font.SysFont("arial", 11)


def shape(surf, kind, col, cx, cy, r, outline=INK, width=2):
    if kind == "circle":
        pygame.draw.circle(surf, col, (cx, cy), r)
        pygame.draw.circle(surf, outline, (cx, cy), r, width)
    elif kind == "square":
        rect = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
        pygame.draw.rect(surf, col, rect)
        pygame.draw.rect(surf, outline, rect, width)
    elif kind == "triangle":
        pts = [(cx, cy - r - 1), (cx - r - 1, cy + r), (cx + r + 1, cy + r)]
        pygame.draw.polygon(surf, col, pts)
        pygame.draw.polygon(surf, outline, pts, width)
    elif kind == "diamond":
        pts = [(cx, cy - r - 2), (cx + r + 2, cy), (cx, cy + r + 2), (cx - r - 2, cy)]
        pygame.draw.polygon(surf, col, pts)
        pygame.draw.polygon(surf, outline, pts, width)
    elif kind == "star":  # player
        pts = []
        for k in range(10):
            ang = -np.pi / 2 + k * np.pi / 5
            rr = r if k % 2 == 0 else r * 0.45
            pts.append((cx + rr * np.cos(ang), cy + rr * np.sin(ang)))
        pygame.draw.polygon(surf, col, pts)
        pygame.draw.polygon(surf, outline, pts, width)


def arrow(surf, col, p0, p1, width=2, dashed=False):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 2:
        return
    u = d / L
    if dashed:
        n = int(L // 8)
        for k in range(0, n, 2):
            pygame.draw.line(surf, col, p0 + u * k * 8, p0 + u * min((k + 1) * 8, L), width)
    else:
        pygame.draw.line(surf, col, p0, p1, width)
    n_ = np.array([-u[1], u[0]])
    pygame.draw.polygon(surf, col, [p1, p1 - u * 8 + n_ * 4, p1 - u * 8 - n_ * 4])


def heat_color(p, pmax):
    v = 0.0 if pmax <= 0 else min(1.0, p / pmax)
    # white -> orange -> dark red (sequential, colour-blind safe enough with the scale bar)
    r = int(255 - 60 * v); g = int(250 - 200 * v); b = int(240 - 230 * v)
    return (r, g, b), 0.15 + 0.7 * v


class Renderer:
    def __init__(self, headless=False):
        init_pygame(headless)
        self.fonts = Fonts()
        self.surf = pygame.Surface((W, H))

    # -------------------------------------------------------------- map area
    def draw_map(self, surf, rect, tr: Trace, t, view="full", sel=0, show_msgs=True, show_heat=False,
                 compact=False, title=None):
        gm = load_map(tr.meta["map"])
        cs = rect.width // SIZE
        ox, oy = rect.x, rect.y
        f = self.fonts
        pygame.draw.rect(surf, BG, rect)

        def cp(c):
            x, y = xy(int(c))
            return ox + x * cs + cs // 2, oy + y * cs + cs // 2

        t = min(t, tr.T)
        npc, pl = tr.positions(t)
        tt = min(t, tr.T - 1)
        vis_sel = tr.vis(tt, sel)
        # cells
        heat = None
        if show_heat and tr.meta.get("has_belief") and "belief" in tr.a:
            heat = tr.a["belief"][tt, sel].astype(float)
            pmax = max(heat.max(), 1e-9)
        for c in range(SIZE * SIZE):
            x, y = xy(c)
            r = pygame.Rect(ox + x * cs, oy + y * cs, cs, cs)
            if gm.wall_flat[c]:
                pygame.draw.rect(surf, WALL, r)
                continue
            col = BG
            if view == "npc" and vis_sel[c]:
                col = (255, 247, 214)
            if heat is not None:
                hc, a = heat_color(heat[c], pmax)
                col = tuple(int(BG[k] * (1 - a) + hc[k] * a) for k in range(3))
            pygame.draw.rect(surf, col, r)
            if not compact:
                pygame.draw.rect(surf, GRID, r, 1)
        if view == "full" and not compact:  # faint outline of the selected NPC's sensor area
            for c in np.flatnonzero(vis_sel):
                x, y = xy(int(c))
                pygame.draw.rect(surf, NPC_COL[sel], pygame.Rect(ox + x * cs, oy + y * cs, cs, cs), 1)
        # doorway ticks
        # messages
        if show_msgs and tr.meta["B"] != "0" and view == "full":  # NPC view never draws hidden positions
            sends = tr.a["sends"][tt] if t < tr.T else np.zeros(tr.N, bool)
            for i in range(tr.N):
                if sends[i]:
                    for j in range(tr.N):
                        if j != i:
                            arrow(surf, NPC_COL[i], cp(npc[i]), cp(npc[j]), 2 if not compact else 1, dashed=True)
            for j, dl in enumerate(tr.meta["delivered"][tt] if t < tr.T else []):
                for (snd, sstep, _tc) in dl:
                    a = np.array(cp(npc[snd]), float); b = np.array(cp(npc[j]), float)
                    arrow(surf, NPC_COL[snd], a + (b - a) * 0.15, a + (b - a) * 0.85, 2 if not compact else 1)
        # NPC-perspective extras
        r = max(cs // 2 - 2, 4)
        if view == "npc" and t < tr.T:
            mc, ms = int(tr.a["mem_cell"][tt, sel]), int(tr.a["mem_step"][tt, sel])
            direct = int(tr.a["direct"][tt, sel])
            if mc >= 0 and direct < 0:  # old sighting drawn as a memory with its age
                p = cp(mc)
                pygame.draw.circle(surf, PLAYER_COL, p, r + 2, 2)
                pygame.draw.line(surf, PLAYER_COL, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), 1)
                pygame.draw.line(surf, PLAYER_COL, (p[0] - r, p[1] + r), (p[0] + r, p[1] - r), 1)
                surf.blit(f.xs.render(f"memory, age {t - ms}", True, PLAYER_COL), (p[0] + r + 2, p[1] - 6))
            for j, c in tr.meta["teammates"][tt][sel]:
                shape(surf, NPC_SHAPE[j], NPC_COL[j], *cp(c), r)
            shape(surf, NPC_SHAPE[sel], NPC_COL[sel], *cp(npc[sel]), r + 2, width=3)
            surf.blit(f.xs.render(str(sel + 1), True, (255, 255, 255)), (cp(npc[sel])[0] - 3, cp(npc[sel])[1] - 6))
            if direct >= 0:
                shape(surf, "star", PLAYER_COL, *cp(direct), r + 3)
            dest = int(tr.a["dests"][tt, sel])
            if dest >= 0:
                p = cp(dest)
                pygame.draw.line(surf, NPC_COL[sel], (p[0], p[1] - r), (p[0], p[1] + r), 2)
                pygame.draw.polygon(surf, NPC_COL[sel], [(p[0], p[1] - r), (p[0] + r, p[1] - r // 2), (p[0], p[1])])
        else:
            for i in range(tr.N):
                shape(surf, NPC_SHAPE[i], NPC_COL[i], *cp(npc[i]), r)
                if not compact:
                    surf.blit(f.xs.render(str(i + 1), True, (255, 255, 255) if i != 1 else INK),
                              (cp(npc[i])[0] - 3, cp(npc[i])[1] - 6))
            shape(surf, "star", PLAYER_COL, *cp(pl), r + 3)
            if t >= tr.T and tr.meta["captured"]:
                pygame.draw.circle(surf, (200, 0, 0), cp(pl), r + 7, 3)
        if title:
            surf.blit(f.b.render(title, True, INK), (ox + 4, oy + 2))
        pygame.draw.rect(surf, (160, 160, 160), rect, 1)
        return cs

    # ----------------------------------------------------------------- panel
    def draw_panel(self, surf, x0, traces, t, view, sel, show_msgs, show_heat, caption=None, speed=None, paused=None):
        f = self.fonts
        y = 10
        tr = traces[0]
        m = tr.meta

        def line(txt, font=None, col=INK, dy=18):
            nonlocal y
            surf.blit((font or f.m).render(txt, True, col), (x0, y))
            y += dy
        line("Cooperative pursuit", f.b, dy=24)
        line(f"View: {'evaluator (all true positions)' if view == 'full' else f'NPC {sel + 1} information only'}",
             f.s, MUTE, 16)
        line("Actors receive only local information.", f.s, MUTE, 20)
        for k, trk in enumerate(traces[:1]):
            line(f"Method: {m['method']}")
            line(f"Player: {m['player_label']}")
            line(f"Protocol: {m['protocol']}", f.s, MUTE, 16)
        tt = min(t, tr.T)
        line(f"Episode {m.get('episode', 1)}   step {tt}/{tr.T}   B = {m['B']}")
        sent = int(tr.a["sent_cum"][min(tt, tr.T) - 1].sum()) if tt > 0 else 0
        allowed = int(tr.a["allowed"][min(tt, tr.T - 1)]) if tt > 0 else 0
        line(f"Packets sent: {sent}   allowed/NPC so far: {allowed}")
        status = "CAPTURED" if (tt >= tr.T and m["captured"]) else ("FAILED (time out)" if tt >= tr.T else "running")
        line(f"Status: {status}", f.m, (190, 0, 0) if status == "CAPTURED" else INK)
        if not m.get("has_belief") and m["method"] != "MARL-OM":
            line("Explicit prediction: absent" if m["method"] == "MARL" else "", f.s, MUTE, 16)
        if m.get("note"):
            line(m["note"], f.xs, MUTE, 14)
        if view == "npc" and tt < tr.T and show_msgs and m["B"] != "0":
            dl = m["delivered"][tt][sel]
            line(f"NPC {sel + 1} received this step: " + (", ".join(f"NPC {a + 1} (sent t={b})" for a, b, _ in dl) or "nothing"),
                 f.xs, MUTE, 15)
        line("Roles now:", f.s, MUTE, 16)
        if tt < tr.T:
            for i in range(tr.N):
                shape(surf, NPC_SHAPE[i], NPC_COL[i], x0 + 8, y + 7, 6, width=1)
                surf.blit(f.s.render(f"NPC {i + 1}: {ROLE_NAMES[int(tr.a['roles'][tt, i])]}", True, INK), (x0 + 22, y))
                y += 17
        else:
            y += 4 * 17
        y += 6
        line("Legend", f.s, MUTE, 16)
        shape(surf, "star", PLAYER_COL, x0 + 8, y + 7, 7, width=1)
        surf.blit(f.s.render("player (target)", True, INK), (x0 + 22, y)); y += 16
        pygame.draw.circle(surf, PLAYER_COL, (x0 + 8, y + 7), 6, 2)
        surf.blit(f.s.render("old sighting (memory + age)", True, INK), (x0 + 22, y)); y += 16
        arrow(surf, (60, 60, 60), (x0, y + 7), (x0 + 16, y + 7), 2, True)
        surf.blit(f.s.render("packet sent (dashed) / delivered (solid)", True, INK), (x0 + 22, y)); y += 16
        pygame.draw.rect(surf, (255, 247, 214), (x0 + 2, y + 2, 12, 12))
        surf.blit(f.s.render("cells visible to selected NPC", True, INK), (x0 + 22, y)); y += 16
        if show_heat:
            if m.get("has_belief") and "belief" in tr.a:
                bar = pygame.Rect(x0, y + 2, 120, 10)
                for k in range(120):
                    c, a = heat_color(k / 119, 1.0)
                    col = tuple(int(BG[q] * (1 - a) + c[q] * a) for q in range(3))
                    pygame.draw.line(surf, col, (bar.x + k, bar.y), (bar.x + k, bar.bottom))
                surf.blit(f.xs.render(f"NPC {sel + 1} belief: low .. high (scaled per frame)", True, INK), (x0, y + 14))
                y += 30
            else:
                surf.blit(f.s.render("heatmap unavailable (no belief recorded)", True, (190, 0, 0)), (x0, y)); y += 18
        if speed is not None:
            line(f"speed {speed:.1f}x {'(paused)' if paused else ''}", f.xs, MUTE, 14)
        y = H - 52
        for s_ in ["Space play/pause  <- -> step  R restart", "1-4 NPC view  V view  M msgs  P belief  +/- speed  Esc"]:
            surf.blit(f.xs.render(s_, True, MUTE), (x0, y)); y += 14
        if caption:
            surf.blit(f.s.render(caption, True, (150, 60, 0)), (x0, H - 22))

    # ------------------------------------------------------------- full frame
    def frame(self, traces, t, view="full", sel=0, show_msgs=True, show_heat=False, caption=None, speed=None,
              paused=None, titles=None):
        s = self.surf
        s.fill((255, 255, 255))
        if len(traces) == 1:
            rect = pygame.Rect(0, 0, 540, 540)
            self.draw_map(s, rect, traces[0], t, view, sel, show_msgs, show_heat)
            self.draw_panel(s, 556, traces, t, view, sel, show_msgs, show_heat, caption, speed, paused)
        else:
            n = len(traces)
            cols = 2
            pw = 360 if n > 2 else 360
            for k, tr in enumerate(traces[:4]):
                px, py = (k % cols) * 370 + 4, (k // cols) * 270
                rect = pygame.Rect(px, py + 22, 260, 240) if False else pygame.Rect(px, py + 18, 240, 240)
                self.draw_map(s, rect, tr, t, view, sel, show_msgs, show_heat, compact=True)
                lbl = (titles[k] if titles else tr.meta["method"])
                s.blit(self.fonts.b.render(lbl, True, INK), (px + 2, py))
                tt = min(t, tr.T)
                status = "CAPTURED" if (tt >= tr.T and tr.meta["captured"]) else ("timed out" if tt >= tr.T else "running")
                sent = int(tr.a["sent_cum"][tt - 1].sum()) if tt > 0 else 0
                s.blit(self.fonts.s.render(f"t={tt}  pkts={sent}  {status}", True, INK), (px + 250 - 0, py + 40))
            self.draw_panel_multi(s, traces, t, caption, speed, paused)
        return s

    def draw_panel_multi(self, s, traces, t, caption, speed, paused):
        f = self.fonts
        x0, y = 750, 10
        s.blit(f.b.render("Side-by-side replay", True, INK), (x0, y)); y += 26
        for ln in ["Same initial scenario and", "player-randomness seed.", "Later player paths may differ",
                   "because the player reacts to", "different pursuers.", "Completed episodes stay marked.",
                   "", "Space play/pause  <- -> step", "R restart  + / - speed  Esc"]:
            s.blit(f.s.render(ln, True, MUTE), (x0, y)); y += 17
        if speed is not None:
            s.blit(f.xs.render(f"speed {speed:.1f}x {'(paused)' if paused else ''}", True, MUTE), (x0, y + 4))
        if caption:
            s.blit(f.s.render(caption, True, (150, 60, 0)), (8, H - 22))

    def to_image(self):
        from PIL import Image
        return Image.frombytes("RGB", (W, H), pygame.image.tostring(self.surf, "RGB"))
