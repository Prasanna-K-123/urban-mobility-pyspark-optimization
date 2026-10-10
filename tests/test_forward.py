import numpy as np
import pandas as pd
import pytest
from benchmark_forward import complete_panel,summarize


def test_zero_inclusive_panel_keeps_inactive_zones_and_dates():
    train=pd.DataFrame({'LocationID':[1,2],'avg_net_vehicle_flow':[3.,-2.]})
    observed=pd.DataFrame({'service_date':['2025-07-01'],'LocationID':[1],'pickups':[1],'dropoffs':[4]})
    p=complete_panel(observed,train,['2025-07-01','2025-07-02'])
    assert len(p)==4 and p.net_flow.tolist()==[3,0,0,0]
    assert p.prediction.tolist()==[3,-2,3,-2]
    s,_=summarize(p);assert s['zero_activity_rows']==3 and s['rows']==4


def test_future_outcomes_cannot_change_frozen_predictions():
    train=pd.DataFrame({'LocationID':[1],'avg_net_vehicle_flow':[4.]})
    observed=pd.DataFrame({'service_date':['2025-07-01'],'LocationID':[1],'pickups':[3],'dropoffs':[5]})
    a=complete_panel(observed,train,['2025-07-01']);observed.dropoffs=100000
    b=complete_panel(observed,train,['2025-07-01'])
    assert a.prediction.equals(b.prediction) and not a.net_flow.equals(b.net_flow)


def test_duplicate_training_or_observed_keys_are_rejected():
    t=pd.DataFrame({'LocationID':[1,1],'avg_net_vehicle_flow':[1,2]})
    o=pd.DataFrame({'service_date':['2025-07-01'],'LocationID':[1],'pickups':[1],'dropoffs':[2]})
    with pytest.raises(ValueError):complete_panel(o,t,['2025-07-01'])
    with pytest.raises(ValueError):complete_panel(pd.concat([o,o]),t.iloc[:1],['2025-07-01'])
