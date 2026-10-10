import pandas as pd
import pytest
from benchmark_forward import build_panel


def test_raw_sql_cleaning_and_hour_counts_against_hand_constructed_records(tmp_path):
    duckdb=pytest.importorskip('duckdb')
    path=tmp_path/'fixture.parquet'
    db=duckdb.connect()
    db.execute('''CREATE TABLE r(tpep_pickup_datetime TIMESTAMP,tpep_dropoff_datetime TIMESTAMP,PULocationID BIGINT,DOLocationID BIGINT,trip_distance DOUBLE)''')
    rows=[('2025-07-01 18:30:00','2025-07-01 18:31:00',1,2,1.),
          ('2025-07-01 14:00:00','2025-07-01 18:00:00',1,2,1.),
          ('2025-06-30 18:30:00','2025-06-30 18:31:00',1,2,1.),
          ('2025-07-01 18:30:00','2025-07-01 18:31:00',1,2,0.),
          ('2025-07-01 18:30:00','2025-07-01 18:31:00',1,2,100.),
          ('2025-07-01 14:00:00','2025-07-01 18:00:01',1,2,1.),
          ('2025-07-01 18:30:00.9','2025-07-01 18:31:00.1',1,2,1.),
          ('2025-07-01 18:30:00','2025-07-01 18:31:00',264,2,1.)]
    db.executemany('INSERT INTO r VALUES(?,?,?,?,?)',rows)
    db.execute('COPY r TO ? (FORMAT PARQUET)',[str(path)]);db.close()
    train=pd.DataFrame({'LocationID':[1,2],'avg_net_vehicle_flow':[0.,0.]})
    panel,meta=build_panel(path,train)
    assert meta['raw_rows']==8 and meta['clean_rows']==2
    day=panel[panel.service_date=='2025-07-01']
    assert day.pickups.tolist()==[1,0] and day.dropoffs.tolist()==[0,2]
    assert day.net_flow.tolist()==[-1,2] and len(panel)==46
