"""Registered complete-panel forecasts with strictly earlier-day features."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reference/forecast_challenge'
FROZEN_PATHS = ('reference/forecast_challenge/PROTOCOL.json', 'src/forecast_challenge.py',
                'collect_forecast_challenge.py', 'benchmark_forecast_challenge.py',
                'verify_forecast_challenge.py', 'requirements-forecast.txt',
                'tests/test_forecast_challenge.py')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def read_protocol():
    return json.loads((OUT / 'PROTOCOL.json').read_text())


def frozen_hashes():
    return {p: sha256(ROOT / p) for p in FROZEN_PATHS}


def panel_bytes(panel):
    return panel.to_csv(index=False, float_format='%.15g', lineterminator='\n').encode()


def write_csv_gzip(path, frame):
    data = panel_bytes(frame)
    Path(path).write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    return data


def read_csv(path):
    # round_trip avoids a parser silently choosing a different last float bit.
    return pd.read_csv(path, float_precision='round_trip')


def complete_counts(counts, zones, start, end):
    dates = pd.bdate_range(start, end).strftime('%Y-%m-%d').tolist()
    zones = sorted(int(z) for z in zones)
    if len(zones) != len(set(zones)):
        raise ValueError('Duplicate frozen zone')
    if counts.duplicated(['service_date', 'LocationID']).any():
        raise ValueError('Duplicate observed zone-date')
    grid = pd.MultiIndex.from_product([dates, zones], names=['service_date', 'LocationID']).to_frame(index=False)
    observed = counts[counts.service_date.isin(dates)].copy()
    outside = observed.loc[~observed.LocationID.isin(zones), ['pickups', 'dropoffs']].sum().astype(int).to_dict()
    panel = grid.merge(observed, how='left', on=['service_date', 'LocationID'], validate='one_to_one')
    panel[['pickups', 'dropoffs']] = panel[['pickups', 'dropoffs']].fillna(0).astype('int64')
    if (panel[['pickups', 'dropoffs']] < 0).any().any():
        raise ValueError('Negative trip count')
    panel['net_flow'] = panel.dropoffs - panel.pickups
    return panel, outside


def reconstruct_panel(paths, zones, start, end):
    """Match original floored-second cleaning, then zero-fill calendar weekdays."""
    db = duckdb.connect()
    try:
        db.execute('SET threads=2')
        db.from_parquet([str(p) for p in paths]).create_view('raw')
        schema = {r[0]: r[1] for r in db.sql('DESCRIBE raw').fetchall()}
        for name in ('tpep_pickup_datetime', 'tpep_dropoff_datetime'):
            if schema.get(name) not in ('TIMESTAMP', 'TIMESTAMP_NS', 'TIMESTAMP_MS'):
                raise ValueError(f'Unexpected wall-clock timestamp schema: {name}={schema.get(name)}')
        raw = int(db.sql('SELECT count(*) FROM raw').fetchone()[0])
        exclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        db.execute('''CREATE TEMP TABLE clean AS
            SELECT tpep_pickup_datetime AS pickup, tpep_dropoff_datetime AS dropoff,
                   PULocationID AS pu, DOLocationID AS dest
            FROM raw WHERE tpep_pickup_datetime IS NOT NULL AND tpep_dropoff_datetime IS NOT NULL
            AND tpep_pickup_datetime >= CAST(? AS TIMESTAMP) AND tpep_pickup_datetime < CAST(? AS TIMESTAMP)
            AND PULocationID BETWEEN 1 AND 263 AND DOLocationID BETWEEN 1 AND 263
            AND trip_distance > 0 AND trip_distance < 100 AND tpep_dropoff_datetime > tpep_pickup_datetime
            AND floor(epoch(tpep_dropoff_datetime - tpep_pickup_datetime)) / 60.0 BETWEEN 1 AND 240''',
                   [start, exclusive_end])
        clean = int(db.sql('SELECT count(*) FROM clean').fetchone()[0])
        counts = db.sql('''WITH p AS (
            SELECT CAST(pickup AS DATE) AS service_date, pu AS LocationID, count(*) AS pickups
            FROM clean WHERE hour(pickup)=18 GROUP BY 1,2), d AS (
            SELECT CAST(dropoff AS DATE) AS service_date, dest AS LocationID, count(*) AS dropoffs
            FROM clean WHERE hour(dropoff)=18 GROUP BY 1,2)
            SELECT coalesce(p.service_date,d.service_date) AS service_date,
                   coalesce(p.LocationID,d.LocationID) AS LocationID,
                   coalesce(p.pickups,0) AS pickups, coalesce(d.dropoffs,0) AS dropoffs
            FROM p FULL JOIN d USING(service_date,LocationID)''').fetchdf()
        counts['service_date'] = pd.to_datetime(counts.service_date).dt.strftime('%Y-%m-%d')
        panel, outside = complete_counts(counts, zones, start, end)
        return panel, {'raw_rows': raw, 'clean_rows': clean, 'weekdays': panel.service_date.nunique(),
                       'zone_days': len(panel), 'zero_activity_rows': int(((panel.pickups == 0) & (panel.dropoffs == 0)).sum()),
                       'outside_frozen_universe_18pm_counts': outside, 'source_schema': schema}
    finally:
        db.close()


def validate_panel(panel):
    panel = panel.sort_values(['service_date', 'LocationID']).reset_index(drop=True)
    if panel.duplicated(['service_date', 'LocationID']).any():
        raise ValueError('Duplicate panel row')
    if panel[['service_date', 'LocationID', 'pickups', 'dropoffs', 'net_flow']].isna().any().any():
        raise ValueError('Missing complete-panel data')
    if not np.array_equal(panel.net_flow, panel.dropoffs - panel.pickups):
        raise ValueError('Net-flow identity failure')
    if not np.equal(panel[['pickups', 'dropoffs']], np.floor(panel[['pickups', 'dropoffs']])).all().all():
        raise ValueError('Noninteger event counts')
    if (panel[['pickups', 'dropoffs']] < 0).any().any():
        raise ValueError('Negative event count')
    dates = sorted(panel.service_date.unique())
    zones = sorted(panel.LocationID.unique())
    expected = pd.MultiIndex.from_product([pd.bdate_range(dates[0], dates[-1]).strftime('%Y-%m-%d'), zones],
                                          names=['service_date', 'LocationID']).to_frame(index=False)
    if not panel[['service_date', 'LocationID']].equals(expected):
        raise ValueError('Missing calendar weekday or frozen zone')
    return panel, dates, zones


def online_features(panel):
    """Generate X before appending the current observed net-flow vector."""
    panel, dates, zones = validate_panel(panel)
    y = panel.net_flow.to_numpy(float).reshape(len(dates), len(zones))
    day_index = pd.DatetimeIndex(dates)
    dow = day_index.dayofweek.to_numpy()
    feature_rows, metadata_rows = [], []
    for i, date in enumerate(day_index):
        same = np.flatnonzero(dow[:i] == dow[i])
        if i < 20 or len(same) < 4:
            continue
        prior, recent = y[:i], y[i - 20:i]
        same4 = y[same[-4:]]
        gap = (date - day_index[i - 1]).days
        history = np.column_stack((prior.mean(axis=0), y[same].mean(axis=0), y[i - 1], y[i - 5],
                                   y[i - 5:i].mean(axis=0), recent.mean(axis=0), np.median(recent, axis=0),
                                   recent.std(axis=0), same4.mean(axis=0), np.median(same4, axis=0)))
        calendar = np.array([float(dow[i] == k) for k in range(5)] +
                            [np.sin(2 * np.pi * date.dayofyear / 365), np.cos(2 * np.pi * date.dayofyear / 365), float(gap)])
        feature_rows.append(np.column_stack((history, np.tile(calendar, (len(zones), 1)))))
        metadata_rows.append(pd.DataFrame({'service_date': dates[i], 'LocationID': zones,
                                           'latest_observation_date': dates[i - 1], 'net_flow': y[i]}))
    if not feature_rows:
        raise ValueError('Insufficient warm-up history')
    X = pd.DataFrame(np.concatenate(feature_rows), columns=read_protocol()['features'])
    metadata = pd.concat(metadata_rows, ignore_index=True)
    if (metadata.latest_observation_date >= metadata.service_date).any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError('Noncausal or incomplete feature')
    return X, metadata


def baseline_predictions(panel, X, metadata):
    training = panel[panel.service_date <= '2025-06-30'].copy()
    training['dow'] = pd.to_datetime(training.service_date).dt.dayofweek
    mean = training.groupby('LocationID').net_flow.mean()
    weekday = training.groupby(['LocationID', 'dow']).net_flow.mean()
    keys = list(zip(metadata.LocationID, pd.to_datetime(metadata.service_date).dt.dayofweek))
    return pd.DataFrame({
        'zero': np.zeros(len(metadata)),
        'frozen_jan_jun_mean': mean.loc[metadata.LocationID].to_numpy(),
        'frozen_jan_jun_weekday_mean': weekday.loc[keys].to_numpy(),
        'seasonal_naive_previous_weekday': X.lag_5_weekdays,
        'rolling_20_weekday_mean': X.prior_20_weekday_mean,
        'rolling_20_weekday_median': X.prior_20_weekday_median,
        'previous_4_same_weekday_mean': X.prior_4_same_weekday_mean,
        'previous_4_same_weekday_median': X.prior_4_same_weekday_median,
    })


def model_candidates(protocol):
    r = protocol['models']['ridge']
    b = protocol['models']['boosting']
    configs = []
    for alpha in r['alphas']:
        configs.append({'name': f'ridge_a{alpha:g}', 'family': 'ridge', 'alpha': alpha})
    for leaves in b['leaf_candidates']:
        configs.append({'name': f'boosting_l{leaves}', 'family': 'boosting', 'max_leaf_nodes': leaves})
    return configs


def estimator(config, protocol):
    if config['family'] == 'ridge':
        p = protocol['models']['ridge']
        return make_pipeline(StandardScaler(), Ridge(alpha=config['alpha'], solver=p['solver'], fit_intercept=p['fit_intercept']))
    p = protocol['models']['boosting']
    return HistGradientBoostingRegressor(max_leaf_nodes=config['max_leaf_nodes'],
                                         **{k: p[k] for k in ('loss', 'max_iter', 'learning_rate', 'min_samples_leaf',
                                                              'l2_regularization', 'early_stopping', 'random_state', 'categorical_features')})


def metrics(y, prediction):
    error = np.asarray(prediction, float) - np.asarray(y, float)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError('Missing/nonfinite prediction')
    return {'rows': len(error), 'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt(np.mean(error ** 2))),
            'signed_bias': float(error.mean()), 'p90_absolute_error': float(np.quantile(np.abs(error), .9))}


def choose(trials, names=None):
    filtered = trials if names is None else [t for t in trials if t['model'] in names]
    if not filtered:
        raise ValueError('No candidate')
    return min(filtered, key=lambda t: (t['mae'], t['model']))['model']


def selection(panel):
    protocol = read_protocol()
    if panel.service_date.max() != protocol['partitions']['selection_end']:
        raise ValueError('Selection must stop at July, never evaluation')
    X, meta = online_features(panel)
    base = baseline_predictions(panel, X, meta)
    fit = meta.service_date.between(protocol['partitions']['fit_start'], protocol['partitions']['fit_end'])
    val = meta.service_date.between(protocol['partitions']['selection_start'], protocol['partitions']['selection_end'])
    trials = []
    prediction = meta[val].reset_index(drop=True).copy()
    for name in protocol['baselines']:
        prediction[name] = base.loc[val, name].to_numpy()
        trials.append({'model': name, 'family': 'baseline', 'training_rows': 0,
                       **metrics(meta.loc[val, 'net_flow'], prediction[name])})
    configs = model_candidates(protocol)
    for config in configs:
        model = estimator(config, protocol).fit(X[fit], meta.loc[fit, 'net_flow'])
        pred = model.predict(X[val])
        prediction[config['name']] = pred
        trials.append({'model': config['name'], 'family': config['family'], 'training_rows': int(fit.sum()),
                       **metrics(meta.loc[val, 'net_flow'], pred)})
    chosen = {family: choose(trials, [c['name'] for c in configs if c['family'] == family]) for family in ('ridge', 'boosting')}
    chosen['advanced'] = choose(trials, [chosen['ridge'], chosen['boosting']])
    chosen['baseline'] = choose(trials, [n for n in protocol['baselines'] if n != 'zero'])
    lock = {'protocol_sha256': sha256(OUT / 'PROTOCOL.json'), 'frozen_code_hashes': frozen_hashes(),
            'selected': chosen, 'selected_configs': {f: next(c for c in configs if c['name'] == chosen[f]) for f in ('ridge', 'boosting')},
            'fit_rows': int(fit.sum()), 'selection_rows': int(val.sum()),
            'final_refit_rows': int(meta.service_date.between(protocol['partitions']['fit_start'], protocol['partitions']['final_refit_end']).sum()),
            'selection_rule': 'July MAE, lexicographic exact ties; inspected development period',
            'development_panel_sha256': sha256(OUT / 'development_panel.csv.gz')}
    return lock, pd.DataFrame(trials), prediction


def evaluate(development, evaluation, lock):
    protocol = read_protocol()
    if lock['protocol_sha256'] != sha256(OUT / 'PROTOCOL.json') or lock['frozen_code_hashes'] != frozen_hashes():
        raise ValueError('Selection protocol/code changed')
    if development.service_date.max() >= evaluation.service_date.min():
        raise ValueError('Evaluation/development overlap')
    panel = pd.concat([development, evaluation], ignore_index=True)
    X, meta = online_features(panel)
    fit = meta.service_date.between(protocol['partitions']['fit_start'], protocol['partitions']['final_refit_end'])
    test = meta.service_date.between(protocol['partitions']['evaluation_start'], protocol['partitions']['evaluation_end'])
    if int(fit.sum()) != lock['final_refit_rows'] or int(test.sum()) != len(evaluation):
        raise ValueError('Final fit/evaluation row budget changed')
    result = meta[test].reset_index(drop=True).copy()
    result = result.merge(evaluation[['service_date', 'LocationID', 'pickups', 'dropoffs']],
                          on=['service_date', 'LocationID'], validate='one_to_one')
    base = baseline_predictions(panel, X, meta)
    for name in protocol['baselines']:
        result[name] = base.loc[test, name].to_numpy()
    for family in ('ridge', 'boosting'):
        config = lock['selected_configs'][family]
        model = estimator(config, protocol).fit(X[fit], meta.loc[fit, 'net_flow'])
        result[config['name']] = model.predict(X[test])
    return result


def paired_uncertainty(prediction, advanced, baseline, protocol):
    difference = np.abs(prediction[advanced] - prediction.net_flow) - np.abs(prediction[baseline] - prediction.net_flow)
    data = prediction[['service_date', 'LocationID']].copy()
    data['difference'] = difference
    iso = pd.to_datetime(data.service_date).dt.isocalendar()
    data['week'] = iso.year.astype(str) + '-' + iso.week.astype(str)
    methods = {}
    for column, seed in (('week', protocol['uncertainty']['week_seed']), ('LocationID', protocol['uncertainty']['zone_seed'])):
        groups = data.groupby(column, sort=True).difference.agg(['sum', 'size'])
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(groups), (protocol['uncertainty']['draws'], len(groups)))
        boot = groups['sum'].to_numpy()[indices].sum(axis=1) / groups['size'].to_numpy()[indices].sum(axis=1)
        methods['iso_week' if column == 'week' else 'zone'] = {'clusters': len(groups), 'draws': len(boot),
                  'seed': seed, 'interval95': np.quantile(boot, [.025, .975]).tolist()}
    return {'mae_difference': float(difference.mean()), 'methods': methods, 'scope': protocol['uncertainty']['limitations']}


def summarize(prediction, development, lock):
    protocol = read_protocol()
    names = protocol['baselines'] + [lock['selected'][f] for f in ('ridge', 'boosting')]
    all_metrics = {n: metrics(prediction.net_flow, prediction[n]) for n in names}
    dates = prediction.service_date
    monthly = {month: {n: metrics(g.net_flow, g[n]) for n in names}
               for month, g in prediction.groupby(dates.str[:7], sort=True)}
    daily_rows = []
    for day, g in prediction.groupby('service_date', sort=True):
        daily_rows.append({'service_date': day, 'rows': len(g), **{n: metrics(g.net_flow, g[n])['mae'] for n in names}})
    daily = pd.DataFrame(daily_rows)
    a, b = lock['selected']['advanced'], lock['selected']['baseline']
    daily['primary_difference'] = daily[a] - daily[b]
    history = development[development.service_date <= '2025-06-30'].copy()
    history['activity'] = history.pickups + history.dropoffs
    activity = history.groupby('LocationID', sort=True).activity.mean().reset_index()
    high = activity.sort_values(['activity', 'LocationID'], ascending=[False, True]).head(int(np.ceil(.2 * len(activity)))).LocationID.tolist()
    strata = {}
    for label, mask in (('high_activity_20pct', prediction.LocationID.isin(high)), ('remaining_80pct', ~prediction.LocationID.isin(high))):
        g = prediction[mask]
        strata[label] = {'zones': g.LocationID.nunique(), 'rows': len(g), 'metrics': {n: metrics(g.net_flow, g[n]) for n in names}}
    summary = {'zone_days': len(prediction), 'weekdays': dates.nunique(), 'zones': prediction.LocationID.nunique(),
               'zero_activity_rows': int(((prediction.pickups == 0) & (prediction.dropoffs == 0)).sum()),
               'selection': lock['selected'], 'model_metrics': all_metrics, 'monthly_metrics': monthly,
               'primary': {'advanced': a, 'baseline': b, **paired_uncertainty(prediction, a, b, protocol),
                           'days_advanced_wins': int((daily.primary_difference < 0).sum()),
                           'days_advanced_loses': int((daily.primary_difference > 0).sum()),
                           'days_tied': int((daily.primary_difference == 0).sum())},
               'high_activity_zones': high, 'activity_strata': strata,
               'availability': protocol['availability'], 'interpretation': protocol['interpretation']}
    return summary, daily


def compare_nested(measured, expected, tolerance=1e-9, path='root'):
    if isinstance(expected, dict):
        if set(measured) != set(expected):
            raise AssertionError(f'{path}: keys differ')
        for k in expected:
            compare_nested(measured[k], expected[k], tolerance, f'{path}.{k}')
    elif isinstance(expected, list):
        if len(measured) != len(expected):
            raise AssertionError(f'{path}: list length differs')
        for i, (a, b) in enumerate(zip(measured, expected)):
            compare_nested(a, b, tolerance, f'{path}[{i}]')
    elif isinstance(expected, float):
        if not np.isclose(measured, expected, atol=tolerance, rtol=0):
            raise AssertionError(f'{path}: {measured} != {expected}')
    elif measured != expected:
        raise AssertionError(f'{path}: {measured} != {expected}')
