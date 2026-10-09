"""Offline structural verification; synthetic cases are not TLC performance."""
from pathlib import Path
import itertools,json
import numpy as np
from src.transport_core import solve_transport,greedy_transport


def exhaustive_2_by_2(s,d,c):
    target=min(sum(s),sum(d)); best=float('inf')
    for values in itertools.product(range(target+1),repeat=4):
        flow=np.asarray(values).reshape(2,2)
        if flow.sum()!=target or np.any(flow.sum(1)>s) or np.any(flow.sum(0)>d): continue
        best=min(best,float(np.sum(flow*c)))
    return best


def main():
    rng=np.random.default_rng(20261009)
    exhaustive=0
    for _ in range(40):
        s,d=rng.integers(0,4,2),rng.integers(0,4,2)
        c=rng.integers(0,11,(2,2)).astype(float)
        _,value=solve_transport(s,d,c)
        assert abs(value-exhaustive_2_by_2(s,d,c))<1e-9
        exhaustive+=1
    gaps=[]
    for _ in range(60):
        s,d=rng.integers(0,30,8),rng.integers(0,30,6)
        c=rng.uniform(0,20,(8,6))
        flow,value=solve_transport(s,d,c)
        greedy,baseline=greedy_transport(s,d,c)
        assert flow.sum()==greedy.sum()==min(s.sum(),d.sum())
        assert value<=baseline+1e-7
        gaps.append(baseline-value)
    summary=dict(seed=20261009,synthetic_cases=100,exhaustive_oracle_cases=exhaustive,
                 capacity_conservation_and_greedy_comparisons=60,
                 all_cases_passed=True,scope='Structural correctness; not a rerun of the 24-million-row TLC analysis.')
    out=Path(__file__).resolve().parent/'results/transport_verification.json'
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
