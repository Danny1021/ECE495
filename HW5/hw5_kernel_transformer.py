#!/usr/bin/env python3
"""
HW5: Kernelized Gradient Descent vs. Fixed-Weights "Transformer" Forward Pass
----------------------------------------------------------------------------
This script demonstrates the equivalence between:
  (A) L steps of kernel gradient descent in function space for squared loss, and
  (B) L layers of a Transformer-like forward pass with fixed (W_Q, W_K, W_V, v)
      that aggregates residuals using the same kernel and applies a residual
      (skip) connection.

What you can do with this script:
- Generate synthetic data in d dimensions with a ground-truth RBF-sum function
- Run both methods and compare predictions across L
- Plot simple 1D visualizations (true f, GD/Transformer approximations)
- Plot error vs L and error vs N curves

Dependencies: numpy, matplotlib
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, List
from pathlib import Path

# --------------------------- Utilities ---------------------------

def set_seed(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)

def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """
    RBF kernel K(x,y) = exp(-||x - y||^2 / gamma)
    X: (N, d), Y: (M, d)
    Returns: (N, M)
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    Xn = np.sum(X * X, axis=1, keepdims=True)  # (N,1)
    Yn = np.sum(Y * Y, axis=1, keepdims=True)  # (M,1)
    # Use broadcasting to get pairwise squared distances
    D2 = Xn + Yn.T - 2 * X @ Y.T
    return np.exp(-D2 / gamma)

# Where to save figures (next to your .py file)
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

def savefig_local(filename: str):
    p = OUT_DIR / filename
    plt.savefig(p, bbox_inches='tight')
    plt.close()
    return str(p)

# --------------------------- Ground Truth f ---------------------------

@dataclass
class RBFSumFunction:
    centers: np.ndarray      # (K, d)
    betas: np.ndarray        # (K,)
    gamma: float             # RBF width parameter

    def __call__(self, X: np.ndarray) -> np.ndarray:
        KX = rbf_kernel(X, self.centers, self.gamma)  # (N,K)
        return KX @ self.betas

def make_ground_truth(d: int, K: int = 5, gamma: float = 0.5, rng: np.random.Generator | None = None) -> RBFSumFunction:
    """
    Create a random sum-of-RBFs function on [-1,1]^d:
      f(x) = sum_{k=1}^K beta_k * exp(-||x - c_k||^2 / gamma)
    """
    if rng is None:
        rng = set_seed(0)
    centers = rng.uniform(-1.0, 1.0, size=(K, d))
    betas = rng.uniform(-1.0, 1.0, size=(K,))
    return RBFSumFunction(centers=centers, betas=betas, gamma=gamma)

# --------------------------- Data Generation ---------------------------

def sample_X(N: int, d: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = set_seed(1)
    return rng.uniform(-1.0, 1.0, size=(N, d))

def make_dataset(N: int, d: int, f_true: Callable[[np.ndarray], np.ndarray], noise_std: float = 0.0, rng: np.random.Generator | None = None):
    if rng is None:
        rng = set_seed(2)
    X = sample_X(N, d, rng)
    y = f_true(X) + rng.normal(0.0, noise_std, size=(N,))
    return X, y

# --------------------------- Kernel GD in Function Space ---------------------------

@dataclass
class KernelGDConfig:
    alpha: float
    gamma: float
    L: int  # number of steps/layers

def kernel_gd_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, cfg: KernelGDConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform L steps of function-space gradient descent for squared loss.
      f_{k+1}(x) = f_k(x) + (alpha/N) sum_i [y_i - f_k(x_i)] k(x_i, x)
    Vectorized over training/query sets.

    Returns:
      f_train: predictions on train after L steps (N,)
      f_query: predictions on query after L steps (M,)
    """
    N = X_train.shape[0]
    K_xx = rbf_kernel(X_train, X_train, cfg.gamma)   # (N,N)
    K_xq = rbf_kernel(X_train, X_query, cfg.gamma)   # (N,M)

    f_train = np.zeros(N)
    f_query = np.zeros(X_query.shape[0])

    for _ in range(cfg.L):
        residual = y_train - f_train             # (N,)
        # update on training points
        f_train = f_train + (cfg.alpha / N) * (K_xx @ residual)
        # update on query points
        f_query = f_query + (cfg.alpha / N) * (K_xq.T @ residual)

    return f_train, f_query

# --------------------------- "Transformer" Forward (Fixed Weights) ---------------------------

def transformer_forward_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, cfg: KernelGDConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Emulates L layers of a Transformer block with:
      - Queries/Keys = identity on x; Value projects residual scalar and scales by -alpha/N
      - Attention weights = kernel k(x_i, x_j) (no softmax)
      - Residual (skip) connection adds updates each layer

    By construction, this exactly matches kernel_gd_predict.
    Returned values mirror kernel_gd_predict for easy comparison.
    """
    N = X_train.shape[0]
    K_xx = rbf_kernel(X_train, X_train, cfg.gamma)   # (N,N)
    K_xq = rbf_kernel(X_train, X_query, cfg.gamma)   # (N,M)

    # "Second coordinate" in the assignment corresponds to negative f; we'll just track f directly
    f_train = np.zeros(N)
    f_query = np.zeros(X_query.shape[0])

    for _ in range(cfg.L):
        # residual = y - f_l(x_i)
        residual = y_train - f_train        # (N,)
        # attention aggregation (values scaled by alpha/N) -> Δf
        delta_train = (cfg.alpha / N) * (K_xx @ residual)  # (N,)
        delta_query = (cfg.alpha / N) * (K_xq.T @ residual)  # (M,)
        # skip connection
        f_train += delta_train
        f_query += delta_query

    return f_train, f_query

# --------------------------- Evaluation ---------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

# --------------------------- Experiments ---------------------------

def experiment_error_vs_L(
    d: int = 10,
    N: int = 60,
    M: int = 200,
    gamma: float = 0.5,
    alpha: float = 0.8,
    L_list: List[int] = [1, 2, 3, 5, 10, 20],
    seed: int = 123,
    noise_std: float = 0.0,
    out_prefix: str = "exp_L"
) -> Dict[str, np.ndarray]:
    """
    Fix N and vary L; compare GD and Transformer predictions vs ground truth on a fresh test set.
    Saves a plot of RMSE vs L.
    """
    rng = np.random.default_rng(seed)
    f_true = make_ground_truth(d=d, K=5, gamma=gamma, rng=rng)
    X_train, y_train = make_dataset(N, d, f_true, noise_std, rng)
    X_test = sample_X(M, d, rng)
    y_test = f_true(X_test)

    gd_errs, tf_errs = [], []
    for L in L_list:
        cfg = KernelGDConfig(alpha=alpha, gamma=gamma, L=L)
        _, gd_pred = kernel_gd_predict(X_train, y_train, X_test, cfg)
        _, tf_pred = transformer_forward_predict(X_train, y_train, X_test, cfg)
        gd_errs.append(rmse(y_test, gd_pred))
        tf_errs.append(rmse(y_test, tf_pred))

    gd_errs = np.array(gd_errs)
    tf_errs = np.array(tf_errs)

    # Plot
    plt.figure()
    plt.plot(L_list, gd_errs, marker='o', label='Kernel GD')
    plt.plot(L_list, tf_errs, marker='x', label='Transformer Forward (fixed)')
    plt.xlabel("Layers / Steps (L)")
    plt.ylabel("RMSE on test set")
    plt.title("Error vs L (N fixed)")
    plt.legend()
    fig_path = savefig_local(f"{out_prefix}_rmse_vs_L.png")

    return {"L": np.array(L_list), "gd_rmse": gd_errs, "tf_rmse": tf_errs, "fig": fig_path}

def experiment_error_vs_N(
    d: int = 10,
    N_list: List[int] = [10, 20, 40, 60, 80, 100],
    M: int = 200,
    gamma: float = 0.5,
    alpha: float = 0.8,
    L: int = 10,
    seed: int = 321,
    noise_std: float = 0.0,
    out_prefix: str = "exp_N"
) -> Dict[str, np.ndarray]:
    """
    Fix L and vary N; compare GD and Transformer predictions vs ground truth.
    Saves a plot of RMSE vs N.
    """
    rng = np.random.default_rng(seed)
    f_true = make_ground_truth(d=d, K=5, gamma=gamma, rng=rng)
    X_test = sample_X(M, d, rng)
    y_test = f_true(X_test)

    gd_errs, tf_errs = [], []
    for N in N_list:
        X_train, y_train = make_dataset(N, d, f_true, noise_std, rng)
        cfg = KernelGDConfig(alpha=alpha, gamma=gamma, L=L)
        _, gd_pred = kernel_gd_predict(X_train, y_train, X_test, cfg)
        _, tf_pred = transformer_forward_predict(X_train, y_train, X_test, cfg)
        gd_errs.append(rmse(y_test, gd_pred))
        tf_errs.append(rmse(y_test, tf_pred))

    gd_errs = np.array(gd_errs)
    tf_errs = np.array(tf_errs)

    # Plot
    plt.figure()
    plt.plot(N_list, gd_errs, marker='o', label='Kernel GD')
    plt.plot(N_list, tf_errs, marker='x', label='Transformer Forward (fixed)')
    plt.xlabel("Training set size (N)")
    plt.ylabel("RMSE on test set")
    plt.title("Error vs N (L fixed)")
    plt.legend()
    fig_path = savefig_local(f"{out_prefix}_rmse_vs_L.png")

    return {"N": np.array(N_list), "gd_rmse": gd_errs, "tf_rmse": tf_errs, "fig": fig_path}

def experiment_1d_visual(
    N: int = 25,
    gamma: float = 0.3,
    alpha: float = 0.8,
    L_list: List[int] = [1, 3, 10],
    seed: int = 999,
    out_prefix: str = "exp_1d"
) -> Dict[str, str]:
    """
    Simple 1D visualization: plot true f and approximations at various L.
    Saves separate figures (one per L) and a combined "all L" plot.
    """
    d = 1
    rng = np.random.default_rng(seed)
    f_true = make_ground_truth(d=d, K=5, gamma=gamma, rng=rng)
    X_train, y_train = make_dataset(N, d, f_true, noise_std=0.0, rng=rng)

    # Grid for plotting
    xx = np.linspace(-1, 1, 400).reshape(-1, 1)
    f_grid = f_true(xx)

    figs = {}

    # Combined plot showing multiple L together
    plt.figure()
    plt.plot(xx[:, 0], f_grid, label="True f")
    for L in L_list:
        cfg = KernelGDConfig(alpha=alpha, gamma=gamma, L=L)
        _, gd_pred_grid = kernel_gd_predict(X_train, y_train, xx, cfg)
        plt.plot(xx[:, 0], gd_pred_grid, label=f"Approx (L={L})")
    plt.scatter(X_train[:, 0], y_train, s=15, label="Train points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("1D: True f and GD approximations")
    plt.legend()
    fig_combined = savefig_local(f"{out_prefix}_1d_allL.png")
    figs["combined"] = fig_combined

    # Individual plots per L
    for L in L_list:
        cfg = KernelGDConfig(alpha=alpha, gamma=gamma, L=L)
        _, gd_pred_grid = kernel_gd_predict(X_train, y_train, xx, cfg)

        plt.figure()
        plt.plot(xx[:, 0], f_grid, label="True f")
        plt.plot(xx[:, 0], gd_pred_grid, label=f"Approx (L={L})")
        plt.scatter(X_train[:, 0], y_train, s=15, label="Train points")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"1D: True f vs Approx (L={L})")
        plt.legend()
        fig_path = savefig_local(f"{out_prefix}_1d_L{L}.png")
        
        figs[f"L{L}"] = fig_path

    return figs

# ---- EXPLICIT transformer forward using h_i^{(ell)} and fixed WQ, WK, WV, v ----
def transformer_forward_explicit(
    X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, cfg: KernelGDConfig
):
    """
    Build hidden states h_i^{(ell)} = [ x_i ; y_i - sum_{k'<=ell} Delta f_{k'}(x_i) ]
    Use fixed WQ, WK to read x-block only; WV to map residual (scalar) to update; v to readout.

    For efficiency, we keep it vectorized and avoid constructing large block matrices,
    but each operation corresponds to the stated WQ, WK, WV, v from the assignment.
    """
    N, d = X_train.shape
    M = X_query.shape[0]

    # Treat train points as tokens 1..N; the query token is N+1
    # Query token has "label" 0 and tracks -f_ell(x_{N+1}) in its scalar coord.
    # We'll compute the same kernel updates as GD.

    K_xx = rbf_kernel(X_train, X_train, cfg.gamma)  # (N,N)  => κ(WQ h_i, WK h_j) for train/train
    K_xq = rbf_kernel(X_train, X_query, cfg.gamma)  # (N,M)  => κ(WQ h_{N+1}, WK h_j)

    # Initialize second coord (residual-like scalars):
    # For train: r_i^(0) = y_i - f_0(x_i) = y_i (since f_0=0)
    # For query: s^(0)  = -f_0(x_{N+1}) = 0
    f_train = np.zeros(N)
    f_query = np.zeros(M)

    for _ in range(cfg.L):
        residual = y_train - f_train  # bottom coord of h_j^{(ell)}
        # WV maps residual -> (-alpha/N)*residual in scalar channel, zeros in x-block
        # Attention sum over j with κ gives Δf
        delta_train = (cfg.alpha / N) * (K_xx @ residual)
        delta_query = (cfg.alpha / N) * (K_xq.T @ residual)

        # residual (skip) adds the update to the scalar coord => f_{ell+1} = f_ell + Δf
        f_train += delta_train
        f_query += delta_query

    # v = [0_d ; -1] so v^T h_{N+1}^{(L)} = f_L(x_{N+1}); we directly return f_query above.
    return f_train, f_query

def experiment_error_vs_L_multi_contexts(
    d=10, N=60, M=200, gamma=0.5, alpha=0.8,
    L_list=[1,2,3,5,10,20], seed=123, noise_std=0.0, num_contexts=10, out_prefix="exp_L_mc"
):
    rng = np.random.default_rng(seed)
    f_true = make_ground_truth(d=d, K=5, gamma=gamma, rng=rng)
    X_test = sample_X(M, d, rng); y_test = f_true(X_test)

    gd_errs = []; tf_errs = []; diffs = []
    for L in L_list:
        cfg = KernelGDConfig(alpha=alpha, gamma=gamma, L=L)
        gd_e, tf_e, df_e = [], [], []
        for _ in range(num_contexts):
            X_train, y_train = make_dataset(N, d, f_true, noise_std, rng)
            _, gd_pred = kernel_gd_predict(X_train, y_train, X_test, cfg)
            _, tf_pred = transformer_forward_explicit(X_train, y_train, X_test, cfg)
            gd_e.append(rmse(y_test, gd_pred))
            tf_e.append(rmse(y_test, tf_pred))
            df_e.append(np.max(np.abs(gd_pred - tf_pred)))  # explicit equality check
        gd_errs.append((np.mean(gd_e), np.std(gd_e)))
        tf_errs.append((np.mean(tf_e), np.std(tf_e)))
        diffs.append((np.mean(df_e), np.max(df_e)))

    # plot means with simple markers
    Lx = np.array(L_list)
    gd_mean = np.array([m for m,s in gd_errs]); tf_mean = np.array([m for m,s in tf_errs])

    plt.figure()
    plt.plot(Lx, gd_mean, marker='o', label='Kernel GD (mean RMSE)')
    plt.plot(Lx, tf_mean, marker='x', label='Transformer (mean RMSE)')
    plt.xlabel("L"); plt.ylabel("RMSE"); plt.title("Error vs L (multi-context, fixed f)")
    fig_path = savefig_local(f"{out_prefix}_rmse_vs_L.png")

    return {"L": Lx, "gd_mean_std": gd_errs, "tf_mean_std": tf_errs, "max_abs_diff_stats": diffs, "fig": fig_path}

def experiment_multiple_functions_summary(
    d=10, N=60, M=200, gamma=0.5, alpha=0.8, L=10,
    num_funcs=5, contexts_per_func=5, seed=777, noise_std=0.0
):
    rng = np.random.default_rng(seed)
    diffs_all = []
    for _ in range(num_funcs):
        f_true = make_ground_truth(d=d, K=5, gamma=gamma, rng=rng)
        X_test = sample_X(M, d, rng); y_test = f_true(X_test)
        for _ in range(contexts_per_func):
            X_train, y_train = make_dataset(N, d, f_true, noise_std, rng)
            cfg = KernelGDConfig(alpha=alpha, gamma=gamma, L=L)
            _, gd_pred = kernel_gd_predict(X_train, y_train, X_test, cfg)
            _, tf_pred = transformer_forward_explicit(X_train, y_train, X_test, cfg)
            diffs_all.append(np.max(np.abs(gd_pred - tf_pred)))
    return {"num_funcs": num_funcs, "contexts_per_func": contexts_per_func,
            "max_abs_diff_mean": float(np.mean(diffs_all)),
            "max_abs_diff_max": float(np.max(diffs_all))}

def experiment_1d_per_layer(N=25, gamma=0.3, alpha=0.8, L=10, seed=999, out_prefix="exp_1d_per_layer_all"):
    d = 1
    rng = np.random.default_rng(seed)
    f_true = make_ground_truth(d=d, K=5, gamma=gamma, rng=rng)
    X_train, y_train = make_dataset(N, d, f_true, 0.0, rng)
    xx = np.linspace(-1, 1, 400).reshape(-1, 1)
    f_grid = f_true(xx)

    Ntr = X_train.shape[0]
    K_xx = rbf_kernel(X_train, X_train, gamma)
    K_xg = rbf_kernel(X_train, xx, gamma)

    f_tr = np.zeros(Ntr)
    f_g = np.zeros(xx.shape[0])

    plt.figure(figsize=(8, 5))
    plt.plot(xx[:, 0], f_grid, 'k--', linewidth=2, label="True f(x)")
    colors = plt.cm.viridis(np.linspace(0, 1, L))

    for k in range(1, L + 1):
        residual = y_train - f_tr
        f_tr = f_tr + (alpha / Ntr) * (K_xx @ residual)
        f_g = f_g + (alpha / Ntr) * (K_xg.T @ residual)
        plt.plot(xx[:, 0], f_g, color=colors[k-1], label=f"f_{k}(x)", alpha=0.8)

    plt.scatter(X_train[:, 0], y_train, s=15, color="black", label="Train points")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("1D Function Evolution Across Layers (k=1..10)")
    plt.legend(ncol=2, fontsize=8)
    fig_path = savefig_local(f"{out_prefix}_all_layers.png")
    return fig_path

def experiment_multiple_functions_plot(
    d=10, N=60, M=200, gamma=0.5, alpha=0.8, L=10,
    num_funcs=10, contexts_per_func=5, seed=777, noise_std=0.0,
    out_prefix="exp_multi_f"
):
    """
    Extension of experiment_multiple_functions_summary:
    Evaluates equivalence between Kernel GD and Transformer across multiple
    random target functions f, then plots a histogram of max abs differences.
    """
    rng = np.random.default_rng(seed)
    diffs_all = []
    for f_idx in range(num_funcs):
        f_true = make_ground_truth(d=d, K=5, gamma=gamma, rng=rng)
        X_test = sample_X(M, d, rng)
        y_test = f_true(X_test)
        for c_idx in range(contexts_per_func):
            X_train, y_train = make_dataset(N, d, f_true, noise_std, rng)
            cfg = KernelGDConfig(alpha=alpha, gamma=gamma, L=L)
            _, gd_pred = kernel_gd_predict(X_train, y_train, X_test, cfg)
            _, tf_pred = transformer_forward_explicit(X_train, y_train, X_test, cfg)
            diffs_all.append(np.max(np.abs(gd_pred - tf_pred)))

    diffs_all = np.array(diffs_all)
    mean_diff, max_diff = np.mean(diffs_all), np.max(diffs_all)

    # Plot histogram of max absolute differences
    plt.figure(figsize=(7,4))
    plt.hist(diffs_all, bins=15, color='skyblue', edgecolor='black')
    plt.axvline(mean_diff, color='red', linestyle='--', label=f"Mean = {mean_diff:.2e}")
    plt.axvline(max_diff, color='orange', linestyle=':', label=f"Max = {max_diff:.2e}")
    plt.xlabel("Max |f_GD(x) - f_Transformer(x)| across test set")
    plt.ylabel("Count (over random f, contexts)")
    plt.title("Equivalence Check Across Multiple Random Functions f")
    plt.legend()
    fig_path = savefig_local(f"{out_prefix}_diff_hist.png")
    plt.close()

    return {
        "num_funcs": num_funcs,
        "contexts_per_func": contexts_per_func,
        "mean_diff": float(mean_diff),
        "max_diff": float(max_diff),
        "fig": fig_path,
    }

# =======================
# Section 7 Visualization
# =======================

def plot_attention_matrix(X_train, gamma=0.5, out_prefix="attention_matrix"):
    """
    Plot the kernel/attention matrix K_ij = kappa(x_i, x_j) for a 2D toy dataset.
    """
    K = rbf_kernel(X_train, X_train, gamma)
    plt.figure(figsize=(5,4))
    im = plt.imshow(K, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Attention / Kernel Matrix  $\\kappa(x_i, x_j)$")
    plt.xlabel("j (keys)"); plt.ylabel("i (queries)")
    fig_path = savefig_local(f"{out_prefix}.png")
    plt.close()
    return fig_path


def plot_residual_updates_2d(f_true, X_train, y_train, gamma=0.5, alpha=0.8,
                             L=5, grid_res=100, out_prefix="residual_updates"):
    """
    Plot Δf_k(x) for k=1..L and the final accumulated f_L(x) on one figure (grid layout).
    """
    # Generate 2D grid
    x1 = np.linspace(-1, 1, grid_res)
    x2 = np.linspace(-1, 1, grid_res)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.stack([X1.ravel(), X2.ravel()], axis=1)

    N = X_train.shape[0]
    K_xx = rbf_kernel(X_train, X_train, gamma)
    K_xg = rbf_kernel(X_train, grid, gamma)

    f_train = np.zeros(N)
    f_grid = np.zeros(grid.shape[0])

    # Create multi-panel plot
    ncols = 3
    nrows = int(np.ceil((L + 1) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 4 * nrows))
    axes = axes.flatten()

    for k in range(1, L + 1):
        residual = y_train - f_train
        delta_train = (alpha / N) * (K_xx @ residual)
        delta_grid = (alpha / N) * (K_xg.T @ residual)
        f_train += delta_train
        f_grid += delta_grid

        ax = axes[k - 1]
        c = ax.contourf(X1, X2, delta_grid.reshape(grid_res, grid_res), 25, cmap="RdBu_r")
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k", s=30)
        ax.set_title(f"Δf{k}(x)")
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)

    # Final accumulated f_L(x)
    ax_final = axes[L]
    c2 = ax_final.contourf(X1, X2, f_grid.reshape(grid_res, grid_res), 25, cmap="RdBu_r")
    ax_final.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k", s=30)
    ax_final.set_title(f"Accumulated f_L(x), L={L}")
    ax_final.set_xlabel("x₁"); ax_final.set_ylabel("x₂")
    fig.colorbar(c2, ax=ax_final, fraction=0.046, pad=0.04)

    # Remove any unused subplots
    for j in range(L + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Residual Updates Δfₖ(x) and Final f_L(x)", fontsize=16, y=1.02)
    plt.tight_layout()
    fig_path = savefig_local(f"{out_prefix}_grid.png")
    plt.close(fig)
    return fig_path

# --------------------------- Main (demo run) ---------------------------

def main_demo():
    d = 10; gamma = 0.5; alpha = 0.8

    # Original plots
    res_L = experiment_error_vs_L(d=d, N=60, M=300, gamma=gamma, alpha=alpha,
                                  L_list=[1,2,3,5,10,20], seed=123, out_prefix="exp_L")
    print("Saved figure (RMSE vs L):", res_L["fig"])

    res_N = experiment_error_vs_N(d=d, N_list=[10,20,40,60,80,100], M=300,
                                  gamma=gamma, alpha=alpha, L=10, seed=321, out_prefix="exp_N")
    print("Saved figure (RMSE vs N):", res_N["fig"])

    # Multi-context (fixed f) + equality check
    mc = experiment_error_vs_L_multi_contexts(num_contexts=10, out_prefix="exp_L_mc")
    print("Max abs diff (GD vs TF) per L (mean, max):", mc["max_abs_diff_stats"])
    print("Saved figure (multi-context):", mc["fig"])

    # Multiple functions generalization check
    gen = experiment_multiple_functions_summary(num_funcs=5, contexts_per_func=5, L=10)
    print("Across multiple f: max_abs_diff mean =", gen["max_abs_diff_mean"], "max =", gen["max_abs_diff_max"])

    # Strict per-layer 1D curves
    experiment_1d_per_layer(N=25, gamma=0.3, alpha=0.8, L=10, out_prefix="exp_1d_per_layer")
    print("Saved 1D per-layer curves in outputs/")

    # Multiple functions generalization (with plot)
    multi_plot = experiment_multiple_functions_plot(
        d=d, N=60, M=300, gamma=gamma, alpha=alpha, L=10,
        num_funcs=10, contexts_per_func=5, seed=777
    )
    print(f"Saved histogram of GD vs Transformer diffs: {multi_plot['fig']}")
    print(f"Mean diff = {multi_plot['mean_diff']:.2e}, Max diff = {multi_plot['max_diff']:.2e}")

        # ---- Section 7 visualizations ----
    d2 = 2
    rng = np.random.default_rng(42)
    f_true2 = make_ground_truth(d=d2, K=5, gamma=0.5, rng=rng)
    X_train2, y_train2 = make_dataset(8, d2, f_true2, 0.0, rng)

    attn_fig = plot_attention_matrix(X_train2, gamma=0.5, out_prefix="attention_matrix")
    print("Saved kernel/attention matrix:", attn_fig)

    resid_fig = plot_residual_updates_2d(f_true2, X_train2, y_train2,
                                         gamma=0.5, alpha=0.8, L=5,
                                         out_prefix="residual_updates")
    print("Saved 2D residual update plots:", resid_fig)



if __name__ == "__main__":
    main_demo()
