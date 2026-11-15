# Evaluator pack: start here

**Question.** Can an explicit local model of the target (opponent modeling) make up for a limited communication budget
when four NPCs pursue one player in a corridor map? Does it still help when the player learns to evade?

## Two-minute walkthrough (nothing to install)
1. Open `slides.pdf` (slide 1: the corridor picture; slides 6-8: results).
2. Open `figures/map_information.png`: the map and what each NPC can see.
3. Watch `demo.gif` (about 100 s, annotated; scenes say whether they are illustrative or recorded).
4. Result plots are in `figures/`; `results_summary.csv` has the numbers behind them.

## Interactive viewer (needs the project environment; CPU only, no checkpoints loaded)
```powershell
.\launch_demo.ps1
```
Controls: Space play/pause, Left/Right step, R restart, 1-4 NPC perspective, V evaluator/NPC view,
M message arrows, P belief heatmap, +/- speed, Tab next replay, Esc exit.

Replays are single recorded examples (not cherry-picked claims of typical behaviour); the plots summarise all tests.
The launcher opens only when you run it and never trains, installs, or uses the internet.
