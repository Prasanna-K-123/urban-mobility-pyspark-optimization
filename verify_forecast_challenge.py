"""Check hashes, reselect/refit, rescore; optionally reconstruct all raw panels."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from collect_forecast_challenge import fetch, build_blocks
from src.forecast_challenge import (ROOT, OUT, read_csv, sha256, selection, evaluate, summarize,
                                    compare_nested, panel_bytes)


def compare_frame(measured, expected, tolerance=1e-8):
    if list(measured.columns) != list(expected.columns) or len(measured) != len(expected):
        raise AssertionError('Frame structure differs')
    maximum = 0.0
    for column in expected.columns:
        a, b = measured[column], expected[column]
        if pd.api.types.is_numeric_dtype(b):
            delta = float(np.max(np.abs(a.to_numpy(float) - b.to_numpy(float)))) if len(a) else 0
            if not np.isfinite(delta) or delta > tolerance:
                raise AssertionError(f'Prediction/count changed: {column}: {delta}')
            maximum = max(maximum, delta)
        elif not a.equals(b):
            raise AssertionError(f'Metadata changed: {column}')
    return maximum


def verify_raw():
    for stage in ('development', 'evaluation'):
        receipt = json.loads((OUT / f'{stage}_sources.json').read_text())
        sources = {s['month']: fetch(s['month'], s['sha256'])[0] for s in receipt['source_files']}
        panel, counts = build_blocks(sources, stage)
        compare_nested(counts, receipt['block_counts'])
        raw = panel_bytes(panel)
        if hashlib.sha256(raw).hexdigest() != receipt['panel_uncompressed_sha256']:
            raise AssertionError('Raw complete-panel reconstruction changed')
        compare_frame(panel, read_csv(OUT / f'{stage}_panel.csv.gz'), tolerance=0)
        print('PINNED_RAW_PANEL_RECONSTRUCTION_PASS', stage, len(panel), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', action='store_true', help='Fetch every pinned Parquet and reconstruct all panels')
    args = parser.parse_args()
    manifest = json.loads((OUT / 'artifact_manifest.json').read_text())
    for path, expected in manifest['files'].items():
        if sha256(ROOT / path) != expected['sha256'] or (ROOT / path).stat().st_size != expected['bytes']:
            raise AssertionError(f'Frozen release bytes changed: {path}')
    if args.raw:
        verify_raw()
    development = read_csv(OUT / 'development_panel.csv.gz')
    lock = json.loads((OUT / 'selection_lock.json').read_text())
    measured_lock, trials, july = selection(development)
    compare_nested(measured_lock, {k: lock[k] for k in measured_lock})
    compare_frame(trials, read_csv(OUT / 'selection_trials.csv'))
    july_delta = compare_frame(july, read_csv(OUT / 'july_selection_predictions.csv.gz'))
    prediction = evaluate(development, read_csv(OUT / 'evaluation_panel.csv.gz'), lock)
    frozen = read_csv(OUT / 'evaluation_predictions.csv.gz')
    delta = compare_frame(prediction, frozen)
    measured, daily = summarize(frozen, development, lock)
    compare_nested(measured, json.loads((OUT / 'summary.json').read_text()))
    compare_frame(daily, read_csv(OUT / 'daily_metrics.csv'))
    print('FORECAST_SELECTION_REFIT_REPLAY_PASS', json.dumps({'july_max_prediction_error': july_delta,
          'evaluation_max_prediction_error': delta, 'raw_reconstruction': args.raw,
          'source_hashes_checked': len(manifest['files'])}), flush=True)


if __name__ == '__main__':
    main()
