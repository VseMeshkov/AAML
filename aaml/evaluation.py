"""
Evaluation tools for comparing learned metrics with BLOSUM/PAM matrices.

Includes:
- Distance matrix comparison (MSE, correlation)
- Heatmap visualization
- Clustering comparison (UPGMA dendrograms)
- Sequence alignment quality comparison
"""

import torch
import numpy as np


def compare_with_blosum(learned_D, name="Learned"):
    """
    Compare a learned distance matrix with BLOSUM62.

    Args:
        learned_D: 20x20 distance tensor (learned)
        name: name for the learned metric

    Returns:
        dict with comparison metrics
    """
    from .loss import _get_default_blosum62

    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    n = len(aa_list)
    idx_map = {aa: i for i, aa in enumerate(aa_list)}

    # Build BLOSUM62 score matrix
    blosum_scores = _get_default_blosum62()
    S_blosum = torch.zeros(n, n)
    for (a, b), score in blosum_scores.items():
        if a in idx_map and b in idx_map:
            i, j = idx_map[a], idx_map[b]
            S_blosum[i, j] = score
            S_blosum[j, i] = score

    # Convert BLOSUM scores to distances
    min_s, max_s = S_blosum.min(), S_blosum.max()
    blosum_D = 1.0 - (S_blosum - min_s) / (max_s - min_s)

    # Ensure learned_D is on CPU for comparison
    if torch.is_tensor(learned_D):
        learned_D = learned_D.detach().cpu()
    else:
        learned_D = torch.tensor(learned_D)

    # Compute comparison metrics
    mse = torch.nn.functional.mse_loss(learned_D, blosum_D).item()

    # Correlation (Pearson) between upper triangular elements
    triu_indices = torch.triu_indices(n, n, offset=1)
    learned_vals = learned_D[triu_indices[0], triu_indices[1]].numpy()
    blosum_vals = blosum_D[triu_indices[0], triu_indices[1]].numpy()

    correlation = np.corrcoef(learned_vals, blosum_vals)[0, 1]
    spearman = _spearman_correlation(learned_vals, blosum_vals)

    # MAE
    mae = np.mean(np.abs(learned_vals - blosum_vals))

    return {
        "name": name,
        "mse": mse,
        "mae": mae,
        "pearson_correlation": correlation,
        "spearman_correlation": spearman,
        "blosum_distance": blosum_D,
        "learned_distance": learned_D
    }


def _spearman_correlation(x, y):
    """Compute Spearman rank correlation."""
    from scipy.stats import rankdata
    rx = rankdata(x)
    ry = rankdata(y)
    return np.corrcoef(rx, ry)[0, 1]


def matrix_to_dataframe(D, aa_list=None):
    """
    Convert a 20x20 distance matrix to a formatted table string.

    Args:
        D: 20x20 matrix (tensor or numpy)
        aa_list: list of AA codes (default: standard 20)

    Returns:
        formatted string table
    """
    if aa_list is None:
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")

    if torch.is_tensor(D):
        D = D.detach().cpu().numpy()

    header = "     " + " ".join(f"{aa:>6}" for aa in aa_list)
    lines = [header]

    for i, aa in enumerate(aa_list):
        row = f"{aa:>3}  " + " ".join(f"{D[i, j]:6.3f}" for j in range(len(aa_list)))
        lines.append(row)

    return "\n".join(lines)


def compute_ranking_correlation(learned_D, blosum_D):
    """
    Compare ranking of amino acid pairs between learned and BLOSUM.

    For each amino acid, rank all other 19 AAs by distance.
    Compare ranks using Kendall's tau.
    """
    from scipy.stats import kendalltau

    if torch.is_tensor(learned_D):
        learned_D = learned_D.detach().cpu().numpy()
    if torch.is_tensor(blosum_D):
        blosum_D = blosum_D.detach().cpu().numpy()

    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    n = len(aa_list)

    tau_values = {}
    for i, aa in enumerate(aa_list):
        # Get distances from this AA to all others
        learned_ranks = np.argsort(np.argsort(learned_D[i]))
        blosum_ranks = np.argsort(np.argsort(blosum_D[i]))

        tau, p = kendalltau(learned_ranks, blosum_ranks)
        tau_values[aa] = tau

    mean_tau = np.mean(list(tau_values.values()))
    return tau_values, mean_tau