# Population and evaluation commands

Generate the permanent 500-match baseline (125 matches for each mode):

```powershell
python scripts/evaluation_control.py generate --kind baseline --root-seed 20260718 --output runs/evaluation/baseline-manifest.json
```

The arena runner must execute the manifest in order with the policy frozen and
write one result per `match_id`. Score it with:

```powershell
python scripts/evaluation_control.py score --manifest runs/evaluation/baseline-manifest.json --results runs/evaluation/baseline-results.json --output runs/evaluation/baseline-report.json
```

Apply the full synthetic population promotion gate:

```powershell
python scripts/evaluation_control.py promotion-gate --results runs/evaluation/promotion-results.json --reference-win-rates runs/evaluation/reference-win-rates.json --output runs/evaluation/promotion-report.json
```

Import exactly 100 held-out candidate-vs-frozen-main outcomes and promote only
when the candidate won at least 60:

```powershell
python scripts/population_control.py --checkpoint-dir checkpoints --candidate checkpoints/candidate.pt --results runs/evaluation/exploiter-results.json --promote
```

Generate and score the disjoint 100-match final human gate:

```powershell
python scripts/evaluation_control.py generate --kind final_human --root-seed 20260718 --output runs/evaluation/final-human-manifest.json
python scripts/evaluation_control.py score --manifest runs/evaluation/final-human-manifest.json --results runs/evaluation/final-human-results.json --output runs/evaluation/final-human-report.json
```

The final command exits successfully only for a frozen, teacher-free run with
at least 95 wins and no losing streak longer than two matches.
