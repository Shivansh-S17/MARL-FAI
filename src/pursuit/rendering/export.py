"""Export recorded traces to an annotated GIF and keyframe PNG/PDF files (headless, shared renderer)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .trace import Trace
from .viewer import Replay, find_replays

FPS = 5
TARGET_FRAMES = 540          # 108 s at 5 fps (required: 90-120 s)
TITLE_FRAMES = 12
HOLD_FRAMES = 8


def _title_card(rd, title, lines, kind):
    import pygame
    from .renderer import H, INK, MUTE, W
    s = rd.surf
    s.fill((255, 255, 255))
    f = rd.fonts
    pygame.draw.rect(s, (240, 240, 236), (60, 120, W - 120, 300), border_radius=8)
    s.blit(pygame.font.SysFont("arial", 30, bold=True).render(title, True, INK), (90, 150))
    y = 215
    for ln in lines:
        s.blit(pygame.font.SysFont("arial", 19).render(ln, True, INK), (90, y)); y += 28
    col = (150, 60, 0) if kind == "illustrative" else (0, 100, 60)
    s.blit(pygame.font.SysFont("arial", 18, bold=True).render(
        "ILLUSTRATIVE example (not an experimental result)" if kind == "illustrative"
        else "Recorded evaluation episode (one example; plots summarise all tests)", True, col), (90, 380))
    return rd.to_image()


def export_demo(a):
    from ..config import ROOT
    from .renderer import Renderer
    src = Path(a.replay or ROOT / "evaluator_pack" / "replays")
    out = Path(a.output or ROOT / "evaluator_pack")
    (out / "figures" / "keyframes").mkdir(parents=True, exist_ok=True)
    rd = Renderer(headless=True)
    scenes = []
    for f in find_replays(src):
        d = json.loads(f.read_text()) if f.suffix == ".json" else {}
        scenes.append((Replay(f), d.get("kind", "experimental"), d.get("card", []), d.get("title", f.stem)))
    if not scenes:
        raise SystemExit("no replays to export")
    budget_per_scene = (TARGET_FRAMES - len(scenes) * (TITLE_FRAMES + HOLD_FRAMES)) // len(scenes)
    frames, key_i = [], 0
    for sc_i, (rp, kind, card, title) in enumerate(scenes):
        frames += [_title_card(rd, f"Scene {sc_i + 1}: {title}", card or [rp.caption or ""], kind)] * TITLE_FRAMES
        stride = max(1, math.ceil((rp.T + 1) / max(budget_per_scene, 1)))
        steps = list(range(0, rp.T + 1, stride))
        if steps[-1] != rp.T:
            steps.append(rp.T)
        cap = rp.caption or ("Illustrative example" if kind == "illustrative" else "")
        if stride > 1:
            cap = (cap + "  " if cap else "") + f"[fast-forward x{stride}]"
        key_steps = {steps[0], steps[len(steps) // 2], steps[-1]}
        for t in steps:
            rd.frame(rp.traces, t, "full", 0, True, any(tr.meta.get("has_belief") for tr in rp.traces) and len(rp.traces) == 1,
                     cap, None, None, rp.titles)
            im = rd.to_image()
            frames.append(im)
            if t in key_steps:
                key_i += 1
                im.save(out / "figures" / "keyframes" / f"keyframe_{key_i:02d}_scene{sc_i + 1}_t{t:03d}.png")
        frames += [frames[-1]] * HOLD_FRAMES
    pal = [im.convert("P", palette=1, colors=96) for im in frames]  # 1 == ADAPTIVE
    dur = 1000 // FPS
    pal[0].save(out / "demo.gif", save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True)
    # keyframe sheet as PDF
    keys = sorted((out / "figures" / "keyframes").glob("keyframe_*.png"))
    if keys:
        from PIL import Image
        ims = [Image.open(k).convert("RGB") for k in keys]
        ims[0].save(out / "figures" / "keyframes.pdf", save_all=True, append_images=ims[1:])
    secs = len(frames) / FPS
    print(f"wrote {out / 'demo.gif'}  frames={len(frames)} duration={secs:.0f}s  keyframes={len(keys)}")
