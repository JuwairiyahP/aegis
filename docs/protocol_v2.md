# AEGIS protocol v2 (frozen 2026-10-27, before any v2 data was collected)

## Why v2
The v1 A/B (`results/`) showed AEGIS eliminating breaches, but replaying the model showed it
did NOT anticipate the surge: in all training runs a fixed 180 s calm phase was always followed
by a ramp and a breach, so the model learned "calm traffic on 1 pod -> breach soon" (the
schedule), and the controller boosted at the start of each run, before the surge existed.
The v1 held-out ramp (45 s) was also too fast for any predictor: load went from safe to
breach in ~26 s, less than the 30 s pod startup.

## Changes
1. **Labels:** a positive label (breach within 120 s) is marked *unforeseeable* and excluded
   from training and evaluation when the surge causing it has not started yet
   (`predictor/dataset.py: mask_unforeseeable`).
2. **Training data** (`loadtest/v2_train.csv`, 14 runs), designed so that only a building surge
   precedes a breach:
   - 4 no-surge runs: flat traffic (4 and 8 req/s) and rises that stay within capacity (16, 20 req/s)
   - 7 surge runs with random calm (240-420 s), base 2-8 req/s, peaks 32-54, ramps 60-240 s,
     2 of them with two surges
   - 3 pre-scaled runs (HPA floor 2, 3, 4), one of them with no surge
   Peaks 42-46 and ramps 150-170 s are excluded from training (held-out region).
   Training set = these 14 runs + v1 EC2 HPA-only runs (et-*, ef-*, xb-*). AEGIS runs (xa-*) are never used.
3. **Deployment gate** (`python -m predictor.diagnose`), on runs not used for training:
   share of calm pre-surge samples with risk >= tau must be <= 10 %, and warnings must come
   after the surge starts. If the gate fails, the model is not deployed and the failure is reported.
4. **Harness** (`harness/run_v2.sh`): in AEGIS runs the controller is switched on 130 s into
   the run, so the predictor's 120 s window contains only this run's calm traffic.
5. **Held-out test / A/B** (`loadtest/v2_ab.csv`): base 4, calm 240 s, peak 44 req/s, ramp 160 s,
   hold 120 s. Three conditions, 4 runs each, order rotated:
   - `yb-*` HPA only (also the model's held-out test set)
   - `ya-*` HPA + AEGIS
   - `ys-*` static over-provisioning: HPA floor fixed at 4 pods (the "just keep more pods" baseline)

## Success criteria (stated before running)
- AEGIS counts as *predictive* only if its boosts start after the surge begins and at least
  30 s before HPA-only runs breach.
- AEGIS is *useful* if it cuts SLO violation vs HPA at clearly lower replica-minutes than static-4.
- Whatever the outcome, all three conditions are reported.
