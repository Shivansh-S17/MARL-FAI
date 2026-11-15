"""Interactive local replay viewer (pygame). Replay mode never imports torch or loads checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

from .trace import Trace


class Replay:
    def __init__(self, path):
        path = Path(path)
        base = path.parent
        if path.suffix == ".json":
            d = json.loads(path.read_text())
            self.title = d.get("title", path.stem)
            self.caption = d.get("caption")
            self.titles = d.get("titles")
            self.traces = [Trace(base / p) for p in d["traces"]]
        else:
            self.traces = [Trace(path)]
            self.title, self.caption, self.titles = path.stem, self.traces[0].meta.get("caption"), None
        self.T = max(tr.T for tr in self.traces)


def find_replays(p):
    p = Path(p)
    if p.is_file():
        return [p]
    files = sorted(p.glob("*.json"))
    files = [f for f in files if "traces" in json.loads(f.read_text())]
    return files or sorted(p.glob("*.npz"))


def run_viewer(a):
    import pygame
    from ..config import ROOT
    from .renderer import H, W, Renderer
    src = a.replay or (ROOT / "evaluator_pack" / "replays")
    files = find_replays(src)
    if not files:
        raise SystemExit(f"no replays found in {src}")
    replays = [Replay(f) for f in files]
    rd = Renderer(headless=False)
    screen = pygame.display.set_mode((W * 3 // 2 if False else 1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("Cooperative pursuit replay viewer")
    clock = pygame.time.Clock()
    idx, t, sel = 0, 0, 0
    view, msgs, heat = "full", True, False
    playing, speed, acc = False, 1.0, 0.0
    run = True
    while run:
        rp = replays[idx]
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                run = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    run = False
                elif k == pygame.K_SPACE:
                    playing = not playing
                elif k == pygame.K_RIGHT:
                    t = min(t + 1, rp.T); playing = False
                elif k == pygame.K_LEFT:
                    t = max(t - 1, 0); playing = False
                elif k == pygame.K_r:
                    t, playing = 0, False
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    sel = k - pygame.K_1
                    view = "npc" if view == "npc" else view
                elif k == pygame.K_v:
                    view = "npc" if view == "full" else "full"
                elif k == pygame.K_m:
                    msgs = not msgs
                elif k == pygame.K_p:
                    heat = not heat
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(speed * 2, 16)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed / 2, 0.25)
                elif k == pygame.K_TAB:
                    idx, t = (idx + 1) % len(replays), 0
        if playing:
            acc += clock.get_time() / 1000.0 * 5 * speed  # 5 recorded transitions per second at 1x
            while acc >= 1 and t < rp.T:
                t += 1; acc -= 1
            if t >= rp.T:
                playing = False
        cap = rp.caption
        fr = rd.frame(rp.traces, t, view, min(sel, rp.traces[0].N - 1), msgs, heat, cap, speed, not playing, rp.titles)
        import pygame as pg
        sw, sh = screen.get_size()
        scale = min(sw / W, sh / H)
        scaled = pg.transform.smoothscale(fr, (int(W * scale), int(H * scale)))
        screen.fill((255, 255, 255))
        screen.blit(scaled, ((sw - scaled.get_width()) // 2, (sh - scaled.get_height()) // 2))
        pg.display.flip()
        clock.tick(30)
    pygame.quit()
