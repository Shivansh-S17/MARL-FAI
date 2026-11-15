# Observation schema

All learned-actor observations come from one adapter, `env/observation.py::build_obs`, which reads only
`LocalInfo` (`controllers/local.py`). `LocalInfo` is fed only by (a) the NPC's own cell and visible cells,
(b) visible agents, (c) delivered packets. It is the single source for controllers, role hints, packet builders and
predictor labels. Update timing at decision t: deliver packets arriving at t -> update memory with direct sighting
(newest observation timestamp wins; direct wins ties) -> predictor labels -> belief -> role hint -> act -> commit.

## Map tensor `[5, 20, 20]` (uint8)
0 known walls, 1 currently visible cells, 2 own location, 3 visible teammate occupancy, 4 visible player location.

## Vector (165 floats; `VEC_DIM`)
| offset | size | content (normalisation) |
| --- | --- | --- |
| 0 | 6 | NPC ID one-hot (6 reserved slots; N=6 needs no schema change) |
| 6 | 1 | remaining time (H-t)/H |
| 7 | 1 | budget B |
| 8 | 1 | available send credit, min(credit,8)/8 |
| 9 | 6 | own previous movement one-hot(5), previous send flag |
| 15 | 10 | latest player sighting: valid, x/19, y/19, age/H, previous displacement one-hot(5), displacement valid |
| 25 | 7 | role one-hot(4), destination valid, destination x/19, y/19 |
| 32 | 126 | six teammate slots x 21: valid, sender x,y, target valid, target x,y, target age/H, displacement valid + one-hot(5), role one-hot(4), destination valid x y, packet age/H |
| 158 | 7 | explicit prediction features (MARL receives zeros): mean x, mean y, var x, var y, cov xy, normalised entropy, max cell probability |
Unknown coordinates carry validity 0 (zero is a legal coordinate). Missing slots are all zero with validity 0.
The previous global reward is never given to the actor.

## Action mask (10 joint actions = movement*2 + send)
Wall moves masked; the send half is masked unless `floor(B*(t+1)) - sent > 0`.

## Never read by actors / role hints / packets / labels
Hidden current player position, remote teammate positions, other NPCs' private history, player type/phase/xi/Q-table/
checkpoint id, centralised training state. Tests: `test_hidden_state_invariance`, `test_actor_input_has_no_player_type_fields`,
`test_role_hint_does_not_use_predictor`.

## Centralised critic (training only; identical for MARL and MARL-OM)
Flattened `[walls, NPC occupancy, player position]` (3x400) + 28 values: remaining time, B, per-NPC sent count and
credit, pending packets, waypoint phase one-hot (zeros for the Q player), Q-player previous action one-hot, Q-player
flag, xi. The critic is removed at execution.

## Predictor (per NPC ID, 42 features)
Previous observed displacement one-hot(5)+validity, coarse target region one-hot(16), own position relative to the
target (2), six teammate slots x (relative x, relative y, visible). Labels: three consecutive direct sightings
(t-2,t-1,t): context at t-1, previous displacement t-2->t-1, label t-1->t. Terminal endpoint labels only if directly
visible. No labels from packets, hidden state, or across gaps.
