# Environment contract

Status: frozen before any learned training (Phase 0). Implemented in `src/pursuit/env/` and checked by `tests/`.

## Geometry and motion
- 20x20 grid, zero-based `(x, y)`, x right, y down. Cell index `c = y*20 + x`.
- Boundary cells are walls. Corridor map: extra walls at x=6 and x=13 for y=1..18 except doorways at y=4, 10, 15
  (six doorways: (6,4) (6,10) (6,15) (13,4) (13,10) (13,15)). Arena map: boundary walls only (secondary study).
- Actions (fixed order): 0 stay, 1 north (y-1), 2 east (x+1), 3 south (y+1), 4 west (x-1). Moves are masked by walls.
- NPCs may share cells and pass through each other; they never block the player.
- All movement is resolved simultaneously. The player chooses from its own pre-transition observation.
- Capture: any NPC-player co-location after the move, or an opposite-direction edge swap. All NPCs terminate together.
- Horizon H = 200 transitions; failure is declared after exactly H transitions. Capture and H expiry are terminals with
  zero bootstrap (time remaining is observable); a PPO rollout boundary is not a terminal.

## Initial placements
Distinct cells uniformly by rejection sampling; standard placements need every NPC-player graph distance >= 4.
Curriculum placements (training only) put every NPC at graph distance 1-3 (first 30k transitions) or 4-8
(next 60k transitions) from the player. All validation/test episodes use standard placements.

## Visibility
`visible(a, b)` iff Manhattan distance <= r (default 5) and the supercover line from cell centre a to b touches no wall.
The supercover includes both cells adjacent to a grid corner, so a wall touched only at a corner blocks sight.
(Tests: `test_visibility_corner_and_wall_blocking`; illustrated in `artifacts/figures/map_information.png`.)

## Communication
Budgets B in {0, 1/8, 1/2, 1}. A send at decision t is allowed iff `sent_i < floor(B*(t+1))` (exact rationals),
so credit starts at zero per episode, accumulates, at most one packet per NPC per step, and the prefix limit
`sent_i(T) <= floor(B*T)` always holds. A packet sent at decision t arrives for decision `t+1+delta`
(delta = 0 by default: no additional delay). Broadcast reaches all other NPCs and is charged once.
Packets are charged even if capture happens on that transition. Payload built from decision-t information only.
Queues, credits, packet slots and target memories reset every episode.

### 16-byte packet (little-endian `<BBHHHBBHHH`)
| bytes | field | notes |
| --- | --- | --- |
| 0 | sender ID | |
| 1 | flags | bit0 sender cell valid, bit1 target valid, bit2 displacement valid, bit3 destination valid, bit4 role valid |
| 2-3 | sender cell | `y*20+x` |
| 4-5 | latest permitted target cell | placeholder 0 with flag cleared |
| 6-7 | target observation step | arrival time never refreshes this timestamp |
| 8 | last valid target displacement action | placeholder 0 with flag cleared |
| 9 | role ID | 0 chase, 1 intercept, 2 block, 3 search |
| 10-11 | intended destination cell | |
| 12-13 | send step | |
| 14-15 | zero padding | |
Application payload cost only (128 bits/packet), not physical headers. No embeddings or extra data are transmitted.

## Rewards
`r = 1[capture] - 0.10/H - 0.01*broadcasts/(N*H) - 0.02*moves/(N*H)`, gamma = 1, GAE lambda 0.95.
Optional training-only shaping: `+0.10*(Phi(s')-Phi(s))`, `Phi = -min_i d(NPC_i, player)/diameter`, `Phi(terminal)=0`.
Evaluation uses capture/duration/communication, never shaped return.

## Players
Waypoint player: route (3,3)->(10,3)->(16,3)->(16,16)->(10,16)->(3,16), shortest path with fixed-order ties, random
initial phase, with probability xi a uniformly random legal action (including stay).
Q-learning player: see `docs/design_decisions.md` and `src/pursuit/opponents/players.py` (tabular, 2160 aggregated states,
epsilon 0.10, alpha 0.10, gamma 0.95, +1/H per surviving step, -1 on capture).

## Randomness
Placements, player draws, NPC action sampling and learning use separate seeded streams. The player's per-time uniform
draws are generated per episode from the scenario's `player_seed` and indexed by time, so changed NPC actions never
shift unrelated streams.
