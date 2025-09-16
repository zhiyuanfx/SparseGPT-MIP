import time
import numpy as np
import torch
import multiprocessing as mp
import gurobipy as gp
from gurobipy import GRB
import solver_config  

_shared_XtX = None
_shared_Xty = None
_shared_nsample = None
_shared_weights = None
_shared_sparsity = None
_shared_n = None
_shared_m = None
_shared_structure = None
_shared_output_dim = None
_shared_input_dim = None

def gurobi_prune(inps: torch.Tensor, outs: torch.Tensor, weights: torch.Tensor, sparsity: float,
                    n: int, m: int, structure: str, dev: torch.device) -> torch.Tensor:
    """
        Run Gurobi pruning row by row, optionally in parallel.
        Returns: 
            pruned_matrix: torch.Tensor with same shape as weights
    """
    if structure not in ["unstructured", "semi"]:
            raise ValueError(f"Unknown sparsity structure: {structure}")
    
    shared_XtX, shared_Xty, shared_nsample = compute_matrices(inps, outs, dev)
    weights_np = weights.cpu().numpy()

    with mp.Pool(
        processes=solver_config.GUROBI_NUM_PROCESS,
        initializer=_init_worker,
        initargs=(shared_XtX, shared_Xty, shared_nsample, weights_np, sparsity, n, m, structure)
    ) as pool:
        results = pool.map(_solve_single_row, range(weights.shape[0]))

    pruned_matrix = torch.zeros_like(weights, dtype=weights.dtype)  
    for row_idx, pruned_matrix_row in results:
        pruned_matrix[row_idx] = pruned_matrix_row.to(weights.dtype)

    return pruned_matrix

def _solve_single_row(row_idx: int) -> tuple[int, torch.Tensor]:
    timed = row_idx <= 0
    if timed:
        start_time = time.time()

    if solver_config.USEWLS:
        env = gp.Env(params=solver_config.params)
    else:
        env = gp.Env()
    model = gp.Model(env=env)
    model.setParam("OutputFlag", 0)
    model.setParam("MIPGap", solver_config.MIPGAP)
    model.setParam("BarConvTol", solver_config.BARCONVTOL)
    model.setParam("Crossover", solver_config.CROSSOVER)
    model.setParam("Method", solver_config.METHOD)
    model.setParam("TimeLimit", solver_config.TIME_LIMIT)

    w = _shared_weights[row_idx]
    input_dim = _shared_input_dim

    z = model.addMVar(input_dim, vtype=GRB.BINARY, name="z")
    e = model.addMVar(input_dim, vtype=GRB.CONTINUOUS, name="e")

    # Objective
    xvar = z * w + e
    loss = (xvar @ (_shared_XtX @ xvar) - 2 * _shared_Xty[:, row_idx] @ xvar) /_shared_nsample
    model.setObjective(loss, GRB.MINIMIZE)

    # Constraints: error bounds
    model.addConstrs((e[i] <=  z[i] * abs(w[i]) * solver_config.EPSILON_PERCENT for i in range(input_dim)))
    model.addConstrs((e[i] >= -z[i] * abs(w[i]) * solver_config.EPSILON_PERCENT for i in range(input_dim)))

    # Sparsity constraints
    if _shared_structure == "unstructured":
        num_nonzero = int(input_dim * _shared_sparsity)
        model.addConstr(z.sum() == num_nonzero)
    elif _shared_structure == "semi":
        for group_start in range(0, input_dim, _shared_m):
            group = z[group_start:group_start + _shared_m]
            if group.shape[0] == _shared_m:
                model.addConstr(group.sum() == _shared_n)
    else:
        raise ValueError(f"Unknown sparsity structure: {_shared_structure}")

    model.optimize()

    if timed:
        print(f"row 0, total time: {time.time() - start_time:.4f}, solving time: {model.Runtime:.4f}")

    z_mask = torch.tensor(z.X > 0.5)
    e_adjust = torch.tensor(e.X)
    pruned_row = torch.tensor(w) * z_mask + e_adjust

    return row_idx, pruned_row

def _init_worker(XtX: np.ndarray, Xty: np.ndarray, nsample: int,
                    weights: np.ndarray, sparsity: float, n: int, m: int, structure: str) -> None:
    global _shared_XtX, _shared_Xty, _shared_nsample, _shared_weights, _shared_sparsity, \
            _shared_n, _shared_m, _shared_structure, _shared_output_dim, _shared_input_dim
    _shared_XtX, _shared_Xty = XtX, Xty
    _shared_nsample = nsample
    _shared_weights = weights
    _shared_sparsity = sparsity
    _shared_n = n
    _shared_m = m
    _shared_structure = structure
    _shared_output_dim, _shared_input_dim = weights.shape

def compute_matrices(inps: torch.Tensor, outs: torch.Tensor, dev: torch.device) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Precompute matrices for the quadratic objective.
    """
    inps = inps.to(dev)
    outs = outs.to(dev)

    XtX = inps.T @ inps
    ridge_lambda = 1e-3 * torch.mean(torch.diag(XtX)).item()
    XtX += ridge_lambda * torch.eye(inps.shape[1], device=dev, dtype=inps.dtype)
    Xty = inps.T @ outs

    if not (torch.isnan(XtX).any() or torch.isinf(XtX).any() or torch.isnan(Xty).any() or torch.isinf(Xty).any()):
        return XtX.cpu().numpy(), Xty.cpu().numpy(), inps.shape[0]

    # If NaN/Inf detected, recompute with float64 and normalization
    print("NaN/Inf detected in XtX. Recomputing with float64 and normalization...")

    inps64 = inps.to(torch.float64)
    outs64 = outs.to(torch.float64)

    inps_std = torch.std(inps64) + 1e-8
    outs_std = torch.std(outs64) + 1e-8
    inps64 = inps64 / inps_std
    outs64 = outs64 / outs_std

    XtX = inps64.T @ inps64
    ridge_lambda = 1e-3 * torch.mean(torch.diag(XtX)).item()
    XtX += ridge_lambda * torch.eye(inps64.shape[1], device=dev, dtype=torch.float64)
    Xty = inps64.T @ outs64
    
    XtX *= inps_std ** 2
    Xty *= inps_std * outs_std

    if torch.isnan(XtX).any() or torch.isinf(XtX).any() or torch.isnan(Xty).any() or torch.isinf(Xty).any():
        raise ValueError("Still NaN/Inf in XtX even after float64 normalization.")

    return XtX.to(inps.dtype).cpu().numpy(), Xty.to(inps.dtype).cpu().numpy(), inps.shape[0]
        