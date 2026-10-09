"""Dependency-light transportation optimizer for replay and invariant checks.

Network-flow constraint matrices are totally unimodular: integer capacities
admit an integer basic optimum without declaring integer decision variables.
The implementation checks integrality and feasibility rather than assuming it.
"""
import numpy as np
from scipy.optimize import linprog


def _validate(supply,demand,cost):
    s,d,c=np.asarray(supply,dtype=float),np.asarray(demand,dtype=float),np.asarray(cost,dtype=float)
    if s.ndim!=1 or d.ndim!=1 or not len(s) or not len(d) or c.shape!=(len(s),len(d)):
        raise ValueError('Invalid transportation dimensions')
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(d)) or not np.all(np.isfinite(c)):
        raise ValueError('Non-finite inputs')
    if np.any(s<0) or np.any(d<0) or np.any(c<0) or np.any(s!=np.rint(s)) or np.any(d!=np.rint(d)):
        raise ValueError('Require nonnegative integer capacities and nonnegative costs')
    return s.astype(int),d.astype(int),c


def solve_transport(supply,demand,cost):
    s,d,c=_validate(supply,demand,cost)
    m,n=c.shape
    target=min(s.sum(),d.sum())
    if not target: return np.zeros((m,n),dtype=int),0.0
    constraints=[]
    for i in range(m):
        a=np.zeros((m,n)); a[i,:]=1; constraints.append(a.ravel())
    for j in range(n):
        a=np.zeros((m,n)); a[:,j]=1; constraints.append(a.ravel())
    fit=linprog(c.ravel(),A_ub=np.asarray(constraints),b_ub=np.r_[s,d],
                A_eq=np.ones((1,m*n)),b_eq=[target],bounds=(0,None),method='highs')
    if not fit.success: raise RuntimeError(fit.message)
    if not np.allclose(fit.x,np.rint(fit.x),atol=1e-7): raise RuntimeError('Non-integer solution')
    flow=np.rint(fit.x).astype(int).reshape(m,n)
    if np.any(flow.sum(axis=1)>s) or np.any(flow.sum(axis=0)>d) or flow.sum()!=target:
        raise RuntimeError('Rounded solution violates capacity/conservation')
    return flow,float(np.sum(flow*c))


def greedy_transport(supply,demand,cost):
    s,d,c=_validate(supply,demand,cost)
    flow=np.zeros(c.shape,dtype=int)
    for i,j in sorted(np.ndindex(c.shape),key=lambda ij:(c[ij],ij)):
        quantity=min(s[i],d[j]); flow[i,j]=quantity; s[i]-=quantity; d[j]-=quantity
    return flow,float(np.sum(flow*c))
