import time
import numpy as np
import torch
import solver_config  
import jax

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "PhiSolve"))
from phisolve.utils.jax_utils import jax_device
from phisolve import PhiMIQP, MIQP, QIHD, JaxAdam, AdamMS

def solve_single_row_qhd(row_idx: int, XtX: np.ndarray, Xty: np.ndarray, weights: np.ndarray, sparsity: float,
                            n: int, m: int, structure: str, nsample: int, device: str) -> np.ndarray:
    """
    Solve a single row pruning problem with QHD.
    Returns:
        pruned_row: numpy array for the pruned row
    """
    start_time = time.time()
    row_len = weights.shape[1]
    w_row = weights[row_idx]

    # M [z;e] -> (z * w + e)
    M = np.hstack((np.diag(w_row), np.eye(row_len)))

    # obj = (1/N) * [(Mw)^T XtX (Mw) - 2 Xty^T (Mw) + yty]
    # obj for MIQP: (1/2) x^T Q x + w^T x
    Q = 2 * M.T @ XtX @ M / nsample
    w = -2 * M.T @ Xty[:, row_idx] / nsample

    # Bounds: z binary, e continuous
    n_vars = 2 * row_len
    lbs = np.zeros(n_vars)
    ubs = np.ones(n_vars)
    lbs[row_len:] = -np.abs(w_row) * solver_config.EPSILON_PERCENT
    ubs[row_len:] = np.abs(w_row) * solver_config.EPSILON_PERCENT

    # Error bounding constraints
    # -z_i * |w_i| eps <= e_i <= z_i * |w_i| eps
    # number of constraints = 2 * row_len
    A = np.zeros((2 * row_len, n_vars))
    b = np.zeros(2 * row_len)

    z_idx = np.arange(row_len)
    e_idx = row_len + np.arange(row_len)

    # first set of constraints:  e_i - z_i*|w_i|*eps <= 0
    A[0:row_len, e_idx] = np.eye(row_len)
    A[0:row_len, z_idx] = -np.diag(np.abs(w_row) * solver_config.EPSILON_PERCENT)

    # second set of constraints: -e_i - z_i*|w_i|*eps <= 0
    A[row_len:, e_idx] = -np.eye(row_len)
    A[row_len:, z_idx] = -np.diag(np.abs(w_row) * solver_config.EPSILON_PERCENT)

    # Structure constraints
    if structure == "unstructured":
        Ceq = np.zeros((1, n_vars))
        Ceq[0, :row_len] = 1
        deq = np.array([int(row_len * sparsity)])
    elif structure == "semi":
        n_groups = row_len // m
        Ceq = np.kron(np.eye(n_groups), np.ones((1, m)))    # shape (n_groups, n_groups*m)
        extra_cols = 2 * row_len - Ceq.shape[1]    # pad so total cols = 2*row_len
        Ceq = np.pad(Ceq, ((0, 0), (0, extra_cols)))
        deq = np.full(Ceq.shape[0], n)
        leftover = row_len % m
        if leftover != 0:
            # force keeping all leftover binary variables
            keep_leftover = np.zeros((1, 2 * row_len))
            keep_leftover[0, row_len - leftover:row_len] = 1
            Ceq = np.vstack((Ceq, keep_leftover))
            deq = np.append(deq, leftover)
    else:
        Ceq, deq = None, None

    prob = MIQP(
        Q, w, A=A, b=b, C=Ceq, d=deq, n_binary_vars=row_len, bounds=(lbs, ubs)
    )
    
    backend = QIHD(
        n_shots=solver_config.QHD_SHOTS, n_steps=solver_config.QHD_STEPS,
        device=device, seed=solver_config.QHD_SEED
    )
    
    refiner = AdamMS(learning_rate=solver_config.ADAMMS_LR, iterations=solver_config.ADAMMS_ITERS, device=device)
    
    model = PhiMIQP(prob, backend, refiner, force_valid_standard='magnitude',
                    sparsity_n=n, sparsity_m=m, sparsity_p=sparsity, sparsity_structure=structure, w_row=w_row)
    res = model.solve()
    
    x_sol = np.array(res.minimizer)
    z_mask = (x_sol[:row_len] > 0.5).astype(float)
    e_adjust = x_sol[row_len:]
    
    pruned_row = w_row * z_mask + e_adjust

    if row_idx == 0:
        print(f"row {row_idx}, total time: {time.time() - start_time:.4f}")
        print(f"minimum: {res.minimum():.6f}, coarse minimum: {res.coarse_min:.6f}")
        print(f"succ_prob (refined): {res.succ_prob():.4f}, succ_prob (coarse): {res.succ_prob_coarse():.4f}")
        print(f"valid percent: {100 * res.percent_in_subspace:.2f}%")
        if res.time is not None:
            print(f"Total runtime: {res.time:.4f} seconds")
        if res.detailed_time is not None:
            print(f"Detailed timing: {res.detailed_time}")

    return pruned_row


def qhd_prune(inps: torch.Tensor, outs: torch.Tensor, weights: torch.Tensor, sparsity: float, 
                n: int, m: int, structure: str, dev: torch.device) -> torch.Tensor:
    """
    Run QHD pruning row by row sequentially.
    Returns:
        pruned_matrix: torch.Tensor with same shape as weights
    """
    if structure not in ["unstructured", "semi"]:
        raise ValueError(f"Unknown sparsity structure: {structure}")

    jax_dev = set_jax_device_from_torch(dev)
    
    XtX, Xty = compute_matrices(inps, outs, dev)
    weights_np = weights.cpu().numpy()
    nsample, _ = inps.shape
    num_rows = weights_np.shape[0]

    pruned_matrix = np.zeros_like(weights_np)

    for row_idx in range(num_rows):
        pruned_matrix[row_idx] = solve_single_row_qhd(
            row_idx, XtX, Xty, weights_np,
            sparsity, n, m, structure, nsample, jax_dev
        )

    return torch.tensor(pruned_matrix, dtype=weights.dtype)

def compute_matrices(inps: torch.Tensor, outs: torch.Tensor, dev: torch.device) -> tuple[np.ndarray, np.ndarray]:
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
        return XtX.cpu().numpy(), Xty.cpu().numpy()

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

    return XtX.to(inps.dtype).cpu().numpy(), Xty.to(inps.dtype).cpu().numpy()

def set_jax_device_from_torch(dev: torch.device) -> str:
    if dev.type == "cpu":
        jax_dev = "cpu"
    elif dev.type.startswith("cuda"):
        jax_dev = "gpu"
    else:
        raise ValueError(f"Unsupported torch device for JAX backend: {dev}")
    jax.config.update("jax_platforms", jax_device(jax_dev))
    return jax_dev
        