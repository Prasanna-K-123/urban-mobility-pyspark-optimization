"""Collect pinned official data in two gated stages; never auto-open final data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import urllib.request

import numpy as np
import pandas as pd

from src.forecast_challenge import ROOT, OUT, read_protocol, sha256, read_csv, reconstruct_panel, write_csv_gzip, frozen_hashes


def fetch(month, expected_sha=None):
    protocol = read_protocol()
    url = protocol['source']['url_template'].format(month=month)
    path = ROOT / 'data' / f'yellow_tripdata_{month}.parquet'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix('.download')
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=45) as response, temporary.open('wb') as output:
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                temporary.replace(path)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                time.sleep(min(attempt + 1, 3))
    digest = sha256(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f'Official source changed: {month}; do not silently update the pin')
    receipt = {'month': month, 'url': url, 'bytes': path.stat().st_size, 'sha256': digest,
               'captured_at': datetime.now(timezone.utc).isoformat()}
    print('SOURCE_COLLECTED', month, receipt['bytes'], digest, flush=True)
    return path, receipt


def build_blocks(sources, stage):
    training = read_csv(ROOT / 'results/zone_imbalance_weekday_6pm.csv')
    zones = sorted(training.LocationID.tolist())
    if len(zones) != 261 or sha256(ROOT / 'results/zone_imbalance_weekday_6pm.csv') != read_protocol()['prior_training_aggregate_sha256']:
        raise ValueError('Original frozen universe/aggregate changed')
    blocks = {}
    panels = []
    if stage == 'development':
        months = [f'2025-{m:02d}' for m in range(1, 7)]
        panel, counts = reconstruct_panel([sources[m] for m in months], zones, '2025-01-01', '2025-06-30')
        mean = panel.groupby('LocationID').net_flow.mean().reindex(training.LocationID).to_numpy()
        if not np.allclose(mean, training.avg_net_vehicle_flow, atol=1e-12, rtol=0):
            raise ValueError('Reconstructed Jan-Jun panel does not match original frozen means')
        if counts['raw_rows'] != 24083384 or counts['clean_rows'] != 22974942 or len(panel) != 33669:
            raise ValueError('Reconstructed original source counts changed')
        blocks['2025-01-through-06'] = counts
        panels.append(panel)
        months = ['2025-07']
    else:
        months = read_protocol()['partitions']['evaluation_months']
    for month in months:
        start = f'{month}-01'
        end = (pd.Timestamp(start) + pd.offsets.MonthEnd()).strftime('%Y-%m-%d')
        panel, counts = reconstruct_panel([sources[month]], zones, start, end)
        if month == '2025-07':
            original = read_csv(ROOT / 'reference/forward_panel/panel.csv')
            if sha256(ROOT / 'reference/forward_panel/panel.csv') != read_protocol()['prior_july_panel_sha256']:
                raise ValueError('Inspected July record changed')
            if not panel.equals(original[panel.columns]):
                raise ValueError('Reconstructed July counts differ from preserved July evidence')
        blocks[month] = counts
        panels.append(panel)
    return pd.concat(panels, ignore_index=True), blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['development', 'evaluation'], required=True)
    parser.add_argument('--registered-commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[a-f0-9]{40}', args.registered_commit):
        raise ValueError('Provide the actually published registration/selection commit')
    protocol = read_protocol()
    stage = args.stage
    receipt_path = OUT / f'{stage}_sources.json'
    if receipt_path.exists():
        raise ValueError('Collection receipt already exists; use immutable verification instead of overwriting')
    if stage == 'evaluation':
        lock = json.loads((OUT / 'selection_lock.json').read_text())
        if lock['protocol_sha256'] != sha256(OUT / 'PROTOCOL.json') or lock['frozen_code_hashes'] != frozen_hashes():
            raise ValueError('Selection lock does not match frozen methods')
        months = protocol['partitions']['evaluation_months']
    else:
        months = protocol['source']['months'][:7]
    sources, receipts = {}, []
    old_july = json.loads((ROOT / 'reference/forward_panel/source.json').read_text())['sha256']
    for month in months:
        sources[month], receipt = fetch(month, old_july if month == '2025-07' else None)
        receipts.append(receipt)
    panel, blocks = build_blocks(sources, stage)
    compressed = OUT / f'{stage}_panel.csv.gz'
    raw = write_csv_gzip(compressed, panel)
    receipt = {'stage': stage, 'registered_commit': args.registered_commit,
               'protocol_sha256': sha256(OUT / 'PROTOCOL.json'), 'source_files': receipts,
               'block_counts': blocks, 'panel_gzip_sha256': sha256(compressed),
               'panel_uncompressed_sha256': __import__('hashlib').sha256(raw).hexdigest(),
               'scope': protocol['availability'], 'duckdb_version': __import__('duckdb').__version__}
    if stage == 'evaluation':
        receipt['selection_lock_sha256'] = sha256(OUT / 'selection_lock.json')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print('COMPLETE_PANEL_COLLECTED', stage, len(panel), json.dumps(blocks), flush=True)


if __name__ == '__main__':
    main()
