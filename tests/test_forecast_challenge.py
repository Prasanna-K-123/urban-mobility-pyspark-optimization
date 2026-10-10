"""Independent small oracles for cleaning, date causality and paired inference."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecast_challenge import (complete_counts, reconstruct_panel, online_features,
                                    validate_panel, choose, metrics, paired_uncertainty,
                                    read_protocol, model_candidates, estimator)


def sequence_panel(length=45):
    days = pd.bdate_range('2025-01-01', periods=length).strftime('%Y-%m-%d')
    rows = []
    for i, day in enumerate(days):
        rows.extend([(day, 1, 0, i + 1, i + 1), (day, 2, i + 1, 0, -i - 1)])
    return pd.DataFrame(rows, columns=['service_date', 'LocationID', 'pickups', 'dropoffs', 'net_flow'])


def test_complete_calendar_zero_fill_and_outside_universe():
    counts = pd.DataFrame([('2025-08-01', 1, 4, 2), ('2025-08-01', 3, 7, 8),
                           ('2025-08-02', 1, 99, 99)],
                          columns=['service_date', 'LocationID', 'pickups', 'dropoffs'])
    panel, outside = complete_counts(counts, [1, 2], '2025-08-01', '2025-08-04')
    assert len(panel) == 4 and panel.net_flow.tolist() == [-2, 0, 0, 0]
    assert outside == {'pickups': 7, 'dropoffs': 8}


def test_duplicate_observation_is_rejected():
    counts = pd.DataFrame([('2025-08-01', 1, 1, 0)] * 2,
                          columns=['service_date', 'LocationID', 'pickups', 'dropoffs'])
    with pytest.raises(ValueError, match='Duplicate'):
        complete_counts(counts, [1], '2025-08-01', '2025-08-04')


@pytest.mark.parametrize('defect', ['missing_day', 'missing_zone', 'wrong_net', 'fractional_count', 'negative_count'])
def test_panel_defects_fail_instead_of_silently_scoring(defect):
    panel = sequence_panel()
    if defect == 'missing_day':
        panel = panel[panel.service_date != '2025-01-06']
    elif defect == 'missing_zone':
        panel = panel.drop(index=11)
    elif defect == 'wrong_net':
        panel.loc[0, 'net_flow'] += 1
    elif defect == 'fractional_count':
        panel[['dropoffs', 'net_flow']] = panel[['dropoffs', 'net_flow']].astype(float)
        panel.loc[0, ['dropoffs', 'net_flow']] = 1.5
    else:
        panel.loc[0, ['dropoffs', 'net_flow']] = -1
    with pytest.raises(ValueError):
        validate_panel(panel)


def test_online_features_match_hand_calculated_sequences():
    panel = sequence_panel()
    X, meta = online_features(panel)
    date = sorted(panel.service_date.unique())[25]
    row = X[(meta.service_date == date) & (meta.LocationID == 1)].iloc[0]
    assert row.prior_expanding_zone_mean == 13
    assert row.lag_1_weekday == 25 and row.lag_5_weekdays == 21
    assert row.prior_5_weekday_mean == 23
    assert row.prior_20_weekday_mean == 15.5 and row.prior_20_weekday_median == 15.5
    assert row.prior_4_same_weekday_mean == 13.5 and row.prior_4_same_weekday_median == 13.5
    expected_dow = pd.Timestamp(date).dayofweek
    assert row[f'dow_{expected_dow}'] == 1 and sum(row[f'dow_{k}'] for k in range(5)) == 1
    assert row.days_since_last_observation == (pd.Timestamp(date) - pd.Timestamp(sorted(panel.service_date.unique())[24])).days
    assert (meta.latest_observation_date < meta.service_date).all()


def test_current_and_all_future_targets_cannot_change_past_features():
    panel = sequence_panel()
    X, meta = online_features(panel)
    cutoff = sorted(panel.service_date.unique())[30]
    altered = panel.copy()
    mask = altered.service_date >= cutoff
    altered.loc[mask, 'dropoffs'] += 1000000
    altered.loc[mask, 'net_flow'] += 1000000
    other, other_meta = online_features(altered)
    assert X[meta.service_date <= cutoff].equals(other[other_meta.service_date <= cutoff])
    assert not X[meta.service_date > cutoff].equals(other[other_meta.service_date > cutoff])


def test_month_and_duration_cleaning_against_separate_record_oracle(tmp_path):
    import duckdb
    path = tmp_path / 'boundary.parquet'
    db = duckdb.connect()
    db.execute('CREATE TABLE r(tpep_pickup_datetime TIMESTAMP,tpep_dropoff_datetime TIMESTAMP,PULocationID BIGINT,DOLocationID BIGINT,trip_distance DOUBLE)')
    records = [
        ('2025-08-01 18:30:00', '2025-08-01 18:31:00', 1, 2, 1.),
        ('2025-08-01 14:00:00', '2025-08-01 18:00:00', 1, 2, 1.),
        ('2025-08-01 14:00:00', '2025-08-01 18:00:01', 1, 2, 1.),
        ('2025-08-01 18:30:00.900', '2025-08-01 18:31:00.100', 1, 2, 1.),
        ('2025-07-31 18:30:00', '2025-07-31 18:31:00', 1, 2, 1.),
        ('2025-08-01 18:30:00', '2025-08-01 18:31:00', 1, 2, 100.),
        ('2025-08-01 18:30:00', '2025-08-01 18:31:00', 1, 2, 0.),
        ('2025-08-01 18:30:00', '2025-08-01 18:31:00', 264, 2, 1.),
        ('2025-08-01 18:30:00', '2025-08-01 18:31:00', 3, 2, 1.),
        ('2025-08-02 18:30:00', '2025-08-02 18:31:00', 1, 2, 1.)]
    db.executemany('INSERT INTO r VALUES(?,?,?,?,?)', records)
    db.execute('COPY r TO ? (FORMAT PARQUET)', [str(path)])
    db.close()
    panel, counts = reconstruct_panel([path], [1, 2], '2025-08-01', '2025-08-04')
    assert counts['raw_rows'] == 10 and counts['clean_rows'] == 4
    assert counts['outside_frozen_universe_18pm_counts'] == {'pickups': 1, 'dropoffs': 0}
    assert panel.pickups.tolist() == [1, 0, 0, 0] and panel.dropoffs.tolist() == [0, 3, 0, 0]
    assert panel.net_flow.tolist() == [-1, 3, 0, 0]


def test_selection_uses_registered_pool_and_exact_tie_break():
    trials = [{'model': 'z', 'mae': 1}, {'model': 'a', 'mae': 1}, {'model': 'unrelated', 'mae': 0}]
    assert choose(trials, ['z', 'a']) == 'a'
    assert choose(trials, ['z']) == 'z'


def test_paired_cluster_intervals_keep_spatial_and_calendar_rows():
    panel = sequence_panel(15)
    panel['advanced'] = panel.net_flow + 2
    panel['baseline'] = panel.net_flow + 4
    result = paired_uncertainty(panel, 'advanced', 'baseline', read_protocol())
    assert result['mae_difference'] == -2
    assert result['methods']['zone']['clusters'] == 2
    for method in result['methods'].values():
        assert method['interval95'] == [-2., -2.]


def test_signed_target_models_allow_negative_predictions_and_three_candidate_budgets():
    protocol = read_protocol()
    configs = model_candidates(protocol)
    assert len(configs) == 6 and sum(c['family'] == 'ridge' for c in configs) == 3
    X = pd.DataFrame({'prior': [-3., -2., -1., 1., 2., 3.]})
    model = estimator(configs[0], protocol).fit(X, X.prior)
    assert model.predict(pd.DataFrame({'prior': [-3.]}))[0] < 0
    assert estimator(configs[-1], protocol).early_stopping is False
    assert metrics([0, 1], [2, 3])['mae'] == 2
