"""Recompute ordinary-source PSNR and paired U-C from the archived test-frame values."""
from pathlib import Path
import csv
import json
import math
from statistics import mean

root = Path(__file__).resolve().parent
with (root / 'summary.csv').open() as stream:
    summary = list(csv.DictReader(stream))
assert len(summary) == 42
sources = json.loads((root / 'sources.json').read_text())
assert len(sources) == 6
for scene in ('vrig_broom', 'vrig_chicken', 'flame_steak'):
    shared_ids = None
    for seed in (0, 1):
        with (root / f'{scene}_seed{seed}_frames.csv').open() as stream:
            frames = list(csv.DictReader(stream))
        target = [r for r in summary if r['scene'] == scene and int(r['seed']) == seed]
        assert len(target) == 7
        values = {}
        for r in target:
            arm, budget = r['arm'], int(r['percent'])
            observations = [x for x in frames if x['arm'] == arm and int(x['budget']) == budget]
            ids = tuple(int(x['source_index']) for x in observations)
            assert len(set(ids)) == len(ids) == int(r['test_frames'])
            if shared_ids is None:
                shared_ids = ids
            assert ids == shared_ids
            score = mean(float(x['psnr_db']) for x in observations)
            assert math.isclose(score, float(r['mean_psnr_db']), rel_tol=0., abs_tol=1e-9)
            assert int(r['kept_points']) == round(int(r['full_points']) * budget / 100)
            values[arm, budget] = score
        deltas = [values['U', q] - values['C', q] for q in (10,15)]
        print(f'{scene:15s} seed {seed}: U-C at 10%, 15% = {deltas[0]:+.3f}, {deltas[1]:+.3f} dB')
