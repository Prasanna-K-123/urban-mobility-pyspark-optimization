"""Locked later-month net-flow forecasting check; no dispatch-profit claim."""
import argparse
import hashlib
import json
import platform
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reference/forward_panel'


def complete_panel(counts,training,dates):
    if not training.LocationID.is_unique or training['avg_net_vehicle_flow'].isna().any():
        raise ValueError('Training predictor rows must be unique and complete')
    grid=pd.MultiIndex.from_product([dates,sorted(training.LocationID)],names=['service_date','LocationID']).to_frame(index=False)
    if counts.duplicated(['service_date','LocationID']).any():raise ValueError('Duplicate observed zone-date')
    panel=grid.merge(counts,on=['service_date','LocationID'],how='left',validate='one_to_one')
    panel[['pickups','dropoffs']]=panel[['pickups','dropoffs']].fillna(0).astype(int)
    panel=panel.merge(training[['LocationID','avg_net_vehicle_flow']],on='LocationID',validate='many_to_one')
    panel['net_flow']=panel.dropoffs-panel.pickups
    panel['prediction']=panel.avg_net_vehicle_flow
    panel['absolute_error']=np.abs(panel.prediction-panel.net_flow)
    panel['zero_baseline_absolute_error']=np.abs(panel.net_flow)
    return panel.drop(columns=['avg_net_vehicle_flow']).sort_values(['service_date','LocationID']).reset_index(drop=True)


def summarize(panel):
    daily=panel.groupby('service_date',sort=True).agg(rows=('LocationID','size'),mae=('absolute_error','mean'),baseline_mae=('zero_baseline_absolute_error','mean'),observed_net=('net_flow','sum'))
    daily['difference']=daily.mae-daily.baseline_mae
    iso=pd.to_datetime(daily.index).isocalendar()
    weeks=iso.year.astype(str)+'-'+iso.week.astype(str)
    unique=weeks.unique();blocks=[daily.difference.to_numpy()[weeks.to_numpy()==w] for w in unique]
    rng=np.random.default_rng(20261010)
    boot=[float(np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))]).mean()) for _ in range(2000)]
    errors=panel.prediction-panel.net_flow
    return dict(rows=len(panel),weekdays=len(daily),zones=panel.LocationID.nunique(),iso_week_blocks=len(blocks),zero_activity_rows=int(((panel.pickups==0)&(panel.dropoffs==0)).sum()),
        frozen_mean=dict(mae=float(panel.absolute_error.mean()),rmse=float(np.sqrt(np.mean(errors**2))),signed_bias=float(errors.mean())),
        zero_baseline=dict(mae=float(panel.zero_baseline_absolute_error.mean()),rmse=float(np.sqrt(np.mean(panel.net_flow**2))),signed_bias=float((-panel.net_flow).mean())),
        mean_minus_zero_mae=float(daily.difference.mean()),paired_iso_week_block_ci95=np.quantile(boot,[.025,.975]).tolist(),bootstrap_draws=2000,
        days_mean_beats_zero=int((daily.difference<0).sum()),days_mean_loses_zero=int((daily.difference>0).sum())),daily.reset_index()


def build_panel(path,training):
    import duckdb
    db=duckdb.connect();db.execute('SET threads=2')
    db.from_parquet(str(path)).create_view('raw')
    raw=int(db.sql('SELECT count(*) FROM raw').fetchone()[0])
    db.execute('''CREATE TEMP TABLE clean AS SELECT tpep_pickup_datetime AS pickup,tpep_dropoff_datetime AS dropoff,PULocationID AS pu,DOLocationID AS dest FROM raw
        WHERE tpep_pickup_datetime IS NOT NULL AND tpep_dropoff_datetime IS NOT NULL
        AND tpep_pickup_datetime>=TIMESTAMP '2025-07-01' AND tpep_pickup_datetime<TIMESTAMP '2025-08-01'
        AND PULocationID BETWEEN 1 AND 263 AND DOLocationID BETWEEN 1 AND 263
        AND trip_distance>0 AND trip_distance<100 AND tpep_dropoff_datetime>tpep_pickup_datetime
        AND floor(epoch(tpep_dropoff_datetime-tpep_pickup_datetime))/60.0 BETWEEN 1 AND 240''')
    clean=int(db.sql('SELECT count(*) FROM clean').fetchone()[0])
    counts=db.sql('''WITH pickups AS (SELECT CAST(pickup AS DATE) AS service_date,pu AS LocationID,count(*) AS pickups FROM clean WHERE hour(pickup)=18 GROUP BY 1,2),
        dropoffs AS (SELECT CAST(dropoff AS DATE) AS service_date,dest AS LocationID,count(*) AS dropoffs FROM clean WHERE hour(dropoff)=18 GROUP BY 1,2)
        SELECT coalesce(p.service_date,d.service_date) AS service_date,coalesce(p.LocationID,d.LocationID) AS LocationID,coalesce(p.pickups,0) AS pickups,coalesce(d.dropoffs,0) AS dropoffs
        FROM pickups p FULL JOIN dropoffs d USING(service_date,LocationID)''').fetchdf()
    counts['service_date']=pd.to_datetime(counts.service_date).dt.strftime('%Y-%m-%d')
    dates=pd.bdate_range('2025-07-01','2025-07-31').strftime('%Y-%m-%d').tolist()
    counts=counts[counts.service_date.isin(dates)].copy()
    outside=counts.loc[~counts.LocationID.isin(training.LocationID),['pickups','dropoffs']].sum().astype(int).to_dict()
    result=complete_panel(counts,training,dates)
    db.close()
    return result,dict(raw_rows=raw,clean_rows=clean,outside_frozen_zone_universe_18pm_counts=outside,duckdb_version=duckdb.__version__)


def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--offline',action='store_true');a=p.parse_args()
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    trainpath=ROOT/'results/zone_imbalance_weekday_6pm.csv'
    assert hashlib.sha256(trainpath.read_bytes()).hexdigest()==protocol['training_aggregate_sha256']
    training=pd.read_csv(trainpath);assert len(training)==261
    if a.offline:
        frozen=json.loads((OUT/'summary.json').read_text());panel=pd.read_csv(OUT/'panel.csv');result,daily=summarize(panel)
        for k,v in result.items():assert v==frozen[k],k
        assert hashlib.sha256((OUT/'panel.csv').read_bytes()).hexdigest()==frozen['panel_sha256']
        assert hashlib.sha256((OUT/'daily_metrics.csv').read_bytes()).hexdigest()==frozen['daily_metrics_sha256']
        print('FORWARD_PANEL_ARTIFACT_REPLAY_PASS (no raw-data reconstruction)');return
    source=json.loads((OUT/'source.json').read_text());path=ROOT/'data/yellow_tripdata_2025-07.parquet'
    if a.download:
        path.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(source['url'],timeout=90) as r,path.open('wb') as f:
            while block:=r.read(4*1024*1024):f.write(block)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
    panel,counts=build_panel(path,training);summary,daily=summarize(panel)
    text=panel.to_csv(index=False,float_format='%.15g');daytext=daily.to_csv(index=False,float_format='%.15g')
    # Scores are recomputed from the committed CSV's precision, so offline
    # artifact replay is exact rather than dependent on hidden extra digits.
    import io
    serialized=pd.read_csv(io.StringIO(text));summary,_=summarize(serialized)
    summary.update(counts,raw_parquet_sha256=source['sha256'],panel_sha256=hashlib.sha256(text.encode()).hexdigest(),daily_metrics_sha256=hashlib.sha256(daytext.encode()).hexdigest(),training_aggregate_sha256=protocol['training_aggregate_sha256'],protocol_sha256=hashlib.sha256((OUT/'PROTOCOL.json').read_bytes()).hexdigest(),environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),scope=protocol['interpretation'])
    if a.verify:
        frozen=json.loads((OUT/'summary.json').read_text())
        # Record the real runner environment without making a Python patch
        # version itself a numerical-reproduction failure. Pinned dependencies,
        # data identity, partitions and all actual scores still must match.
        measured={k:v for k,v in summary.items() if k!='environment'}
        expected={k:v for k,v in frozen.items() if k!='environment'}
        assert measured==expected,'Raw forward reconstruction differs from frozen release'
        assert text==(OUT/'panel.csv').read_text(),'Raw reconstructed panel differs'
        print('RUNNER_ENVIRONMENT='+json.dumps(summary['environment']))
        print('FORWARD_RAW_DATA_RECONSTRUCTION_PASS')
    else:
        (OUT/'panel.csv').write_text(text);(OUT/'daily_metrics.csv').write_text(daytext);(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('FORWARD_SUMMARY='+json.dumps(summary))


if __name__=='__main__':main()
