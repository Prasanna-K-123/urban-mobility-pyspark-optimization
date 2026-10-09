import numpy as np
import pytest
from src.transport_core import solve_transport,greedy_transport


def test_global_optimum_avoids_greedy_trap():
    c=np.array([[1,2],[2,100]])
    flow,cost=solve_transport([1,1],[1,1],c)
    _,greedy=greedy_transport([1,1],[1,1],c)
    assert cost==4 and greedy==101
    assert np.array_equal(flow,[[0,1],[1,0]])


def test_insufficient_supply_and_zero_flow():
    flow,cost=solve_transport([0,2],[3,4],[[1,2],[3,4]])
    assert flow.sum()==2 and np.all(flow.sum(0)<=[3,4])
    flow,cost=solve_transport([0],[4],[[5]])
    assert flow.sum()==0 and cost==0


def test_fractional_capacity_and_nonfinite_cost_rejected():
    with pytest.raises(ValueError): solve_transport([.5],[1],[[2]])
    with pytest.raises(ValueError): solve_transport([1],[1],[[np.nan]])
