import torch
import time
import multiprocessing as mp
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import gurobi_config

EPSILON_PERCENT = 1    # |compensation| < epsilon_percent * |original_weight|

def solve_mask(inps, outs, weights, sparsity, n, m, method='gurobi', structure='unstructured', **kwargs):
    """
    Dispatch function to solve for binary pruning mask using specified method.

    Args:
        inps: [num_samples, input_dim] calibration inputs
        outs: [num_samples, output_dim] corresponding outputs from dense layer
        weights: [output_dim, input_dim] original dense weights
        n: number of zeros per block
        m: block size (e.g., 4 for 2:4)
        method: which solver to use (e.g., 'gurobi')
        **kwargs: optional args like env for gurobi

    Returns:
        mask: Boolean tensor of same shape as weights
    """
    if method == 'gurobi':
        return solve_with_gurobi_parallel(inps, outs, weights, sparsity, n, m, structure, **kwargs)
    else:
        raise ValueError('Unknown mask solving method: ' + method)

_shared_inps = None
_shared_outs = None
_shared_weights = None
_shared_n = None
_shared_m = None
_shared_structure = None
_shared_sparsity = None

def _init_worker(inps_np, outs_np, weights_np, sparsity, n, m, structure):
    global _shared_inps, _shared_outs, _shared_weights, _shared_n, _shared_m, _shared_structure, _shared_sparsity
    _shared_inps = inps_np
    _shared_outs = outs_np
    _shared_weights = weights_np
    _shared_sparsity = sparsity
    _shared_n = n
    _shared_m = m
    _shared_structure = structure

def solve_single_row(row_idx):
    timed = row_idx <= 0
    if timed:
        start_time = time.time()

    env = gp.Env(params=gurobi_config.params)
    model = gp.Model(env=env)
    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", gurobi_config.MIPGAP)
    model.setParam("BarConvTol", gurobi_config.BARCONVTOL)
    model.setParam("Crossover", gurobi_config.CROSSOVER)
    model.setParam("Method", gurobi_config.METHOD) 
    model.setParam("TimeLimit", gurobi_config.TIME_LIMIT)
  
    inp = _shared_inps
    out = _shared_outs[:, row_idx]
    w = _shared_weights[row_idx]
    n = _shared_n
    m = _shared_m
    structure = _shared_structure
    sparsity = _shared_sparsity
    input_dim = inp.shape[1]

    z = model.addMVar(input_dim, vtype=GRB.BINARY, name="z")
    e = model.addMVar(input_dim, vtype=GRB.CONTINUOUS, name="e")
    
    AtA = inp.T @ inp 
    ridge_lambda = 1e-3 * np.mean(np.diag(AtA)) 
    AtA += ridge_lambda * np.eye(input_dim)
    Atb = inp.T @ out
    btb = out @ out 

    xvar = z * w + e
    loss = (xvar @ (AtA @ xvar) - 2 * Atb @ xvar + btb) / inp.shape[0]

    model.setObjective(loss, GRB.MINIMIZE)
    
    model.addConstrs((e[i] <=  z[i] * abs(w[i]) * EPSILON_PERCENT for i in range(input_dim)))
    model.addConstrs((e[i] >= -z[i] * abs(w[i]) * EPSILON_PERCENT for i in range(input_dim)))

    if structure == "unstructured":
        # unstructured sparsity
        num_nonzero = int(input_dim * sparsity)
        model.addConstr(z.sum() == num_nonzero)
    elif structure == "semi":
        # n : m semi-structured sparsity
        for group_start in range(0, input_dim, m):
            group = z[group_start:group_start + m]
            if group.shape[0] == m:
                model.addConstr(group.sum() == n)
    else:
        raise ValueError(f"Unknown sparsity structure: {structure}")

    model.optimize()
    
    if timed:
        print(f"1st row, total time: {time.time() - start_time:.4f}, solving time: {model.Runtime:.4f}")
        
    z_mask = torch.tensor(z.X > 0.5)
    e_adjust = torch.tensor(e.X)
    pruned_row = torch.from_numpy(w) * z_mask + e_adjust
    
    return row_idx, pruned_row

def solve_with_gurobi_parallel(inps, outs, weights, sparsity, n, m, structure="unstructured"):

    inps_np = inps.cpu().numpy()
    outs_np = outs.cpu().numpy()
    weights_np = weights.cpu().numpy()
    rows = weights.shape[0]

    with mp.Pool(
        processes=2,
        initializer=_init_worker,
        initargs=(inps_np, outs_np, weights_np, sparsity, n, m, structure)
    ) as pool:
        results = pool.map(solve_single_row, range(rows))

    pruned_matrix = torch.zeros_like(weights)  
    for row_idx, pruned_matrix_row in results:
        pruned_matrix[row_idx] = pruned_matrix_row.to(weights.dtype)

    return pruned_matrix