"""July selection first, then a separate three-month evaluation; no test tuning."""
import argparse
from datetime import datetime, timezone
import json
import platform

import numpy as np
import pandas as pd
import sklearn

from src.forecast_challenge import (ROOT, OUT, FROZEN_PATHS, read_csv, sha256, selection,
                                    evaluate, summarize, write_csv_gzip, panel_bytes)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--select', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.select == args.evaluate:
        parser.error('Choose exactly one stage')
    development = read_csv(OUT / 'development_panel.csv.gz')
    if args.select:
        if (OUT / 'selection_lock.json').exists():
            raise ValueError('Already frozen; verify instead of overwriting selection')
        if any((ROOT / 'data' / f'yellow_tripdata_2025-{m:02d}.parquet').exists() for m in (8, 9, 10)):
            raise ValueError('Evaluation file appeared before selection was locked')
        lock, trials, prediction = selection(development)
        write_csv_gzip(OUT / 'july_selection_predictions.csv.gz', prediction)
        (OUT / 'selection_trials.csv').write_bytes(panel_bytes(trials))
        lock['created_at'] = datetime.now(timezone.utc).isoformat()
        lock['selection_trials_sha256'] = sha256(OUT / 'selection_trials.csv')
        lock['selection_predictions_sha256'] = sha256(OUT / 'july_selection_predictions.csv.gz')
        write_json(OUT / 'selection_lock.json', lock)
        print('JULY_DEVELOPMENT_SELECTION_LOCKED', json.dumps(lock), flush=True)
        print(trials.to_string(index=False), flush=True)
        return
    if (OUT / 'summary.json').exists():
        raise ValueError('Evaluation already frozen; replay instead of rewriting')
    lock = json.loads((OUT / 'selection_lock.json').read_text())
    receipt = json.loads((OUT / 'evaluation_sources.json').read_text())
    if receipt['selection_lock_sha256'] != sha256(OUT / 'selection_lock.json'):
        raise ValueError('Evaluation source collection preceded or differs from locked selection')
    final = read_csv(OUT / 'evaluation_panel.csv.gz')
    prediction = evaluate(development, final, lock)
    write_csv_gzip(OUT / 'evaluation_predictions.csv.gz', prediction)
    # Re-score the actual committed precision, not an unpublished array.
    prediction = read_csv(OUT / 'evaluation_predictions.csv.gz')
    summary, daily = summarize(prediction, development, lock)
    write_json(OUT / 'summary.json', summary)
    (OUT / 'daily_metrics.csv').write_bytes(panel_bytes(daily))
    write_json(OUT / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
                                        'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                                        'duckdb': __import__('duckdb').__version__})
    paths = list(FROZEN_PATHS) + [
        'results/zone_imbalance_weekday_6pm.csv', 'reference/forward_panel/panel.csv',
        *[f'reference/forecast_challenge/{n}' for n in ('development_sources.json', 'evaluation_sources.json',
         'development_panel.csv.gz', 'evaluation_panel.csv.gz', 'july_selection_predictions.csv.gz',
         'selection_trials.csv', 'selection_lock.json', 'evaluation_predictions.csv.gz', 'summary.json',
         'daily_metrics.csv', 'environment.json')]]
    write_json(OUT / 'artifact_manifest.json', {'files': {p: {'sha256': sha256(ROOT / p), 'bytes': (ROOT / p).stat().st_size} for p in paths},
               'scope': 'Frozen protocol/methods/data/predictions/summary; manifest excludes itself and review prose.'})
    print('REGISTERED_THREE_MONTH_EVALUATION', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
