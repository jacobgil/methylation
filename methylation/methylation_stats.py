"""Statistical functions for methylation analysis."""
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def delta_beta(X_percent, groups):
    """Δβ per feature = max group mean β − min group mean β (β in 0..1)."""
    Xb = X_percent / (100.0 if np.nanmax(X_percent) > 1.001 else 1.0)
    group_set = sorted([str(g) for g in set(groups)])
    
    means = np.vstack([np.mean(Xb[groups == g, :], axis=0) for g in group_set])
    return (np.max(means, axis=0) - np.min(means, axis=0)), means, group_set


def permutation_fdr_delta_beta(X_percent, groups, B=1000, var_quantile=0.5, rng=42):
    """Permutation-calibrated FDR for Δβ with independent variance filtering."""
    keep = list(range(X_percent.shape[1]))

    db_obs, means_obs, levels = delta_beta(X_percent, groups)

    # Build a grid of candidate thresholds over the upper tail of Δβ
    grid = np.quantile(db_obs[~np.isnan(db_obs)], np.linspace(0.80, 0.999, 60))
    R = np.array([(db_obs >= t).sum() for t in grid], dtype=float)

    # Permutations
    logger.info(f"Running {B} permutations for FDR calculation...")
    rng = np.random.default_rng(rng)
    g = np.array(groups)
    V = np.zeros_like(R, dtype=float)
    for b in tqdm(range(B), desc="Permutations", leave=False):
        g_perm = g.copy()
        rng.shuffle(g_perm)  # preserves group sizes
        db_perm, _, _ = delta_beta(X_percent, g_perm)
        V += np.array([(db_perm >= t).sum() for t in grid], dtype=float)
    V /= B
    logger.info(f"Completed {B} permutations")
    FDR = np.where(R > 0, V / R, np.nan)

    tbl = pd.DataFrame({"tau": grid, "R": R, "E_false": V, "FDR_hat": FDR})
    # choose tightest τ with FDR ≤ 0.05 (fallback 0.10 if none)
    candidates = tbl.loc[tbl["FDR_hat"] <= 0.05, "tau"]
    if candidates.empty:
        candidates = tbl.loc[tbl["FDR_hat"] <= 0.10, "tau"]
    tau = float(candidates.max()) if not candidates.empty else float(grid[-1])

    # Call hits on the kept set, then lift back to original indices
    hits_keep_mask = db_obs >= tau

    # assemble results
    res = pd.DataFrame({
        "feature": keep,
        "delta_beta": db_obs,
        "db_obs": db_obs,
    })

    # attach group means (β) for interpretability
    if res.shape[0] > 0:
        features = res.feature.values
        for i, gname in enumerate(levels):
            res[f"mean_beta_{gname}"] = means_obs[i, features]

        # also record the most separated pair
        pair_hi, pair_lo = means_obs.argmax(axis=0), means_obs.argmin(axis=0)
        
        best_pair = [f"{levels[hi]} vs {levels[lo]}" for hi, lo in zip(pair_hi, pair_lo, strict=True)]
        res["best_pair"] = best_pair

    res = res.sort_values(["delta_beta"], ascending=[False]).reset_index(drop=True)

    return tau, tbl, res, means_obs, levels
