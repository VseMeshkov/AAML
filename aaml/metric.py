"""
Learnable distance metric (Mahalanobis) on the amino acid feature space.

d(a, b) = tanh((x_a - x_b)^T M (x_a - x_b))

where M = L^T L is a positive semi-definite matrix ensuring the
distance satisfies metric properties (symmetry, non-negativity).
"""

import torch
import torch.nn as nn


class AA_Metric(nn.Module):
    """
    Learnable Mahalanobis distance metric on amino acid feature space.

    The metric is defined as: d(a,b) = tanh((x_a - x_b)^T M (x_a - x_b))
    where M = L^T L is PSD, ensuring d(a,b) >= 0 and d(a,b) = d(b,a).
    """

    def __init__(self, feature_space, d=5, init_identity=True):
        super().__init__()
        self.feature_space = feature_space
        self.d = d  # feature dimension (default 5)

        # Parameterize M = L^T L for PSD constraint
        # L is lower triangular with diagonal log-parameterization
        self._L_diag_log = nn.Parameter(torch.zeros(d))
        self._L_offdiag = nn.Parameter(torch.zeros(d, d))

        if init_identity:
            # Initialize L as identity matrix
            nn.init.constant_(self._L_diag_log, 0.0)  # exp(0) = 1 on diagonal
            nn.init.zeros_(self._L_offdiag)

        # Cache for asymmetry penalty
        self._aa_list = list("ACDEFGHIKLMNPQRSTVWY")

    def get_L(self):
        """Construct lower triangular matrix L from parameters."""
        diag = torch.exp(self._L_diag_log)  # ensure positive diagonal
        L = torch.tril(self._L_offdiag)
        # Set diagonal
        L = L + torch.diag(diag)
        return L

    def get_M(self):
        """Compute M = L^T L (PSD matrix)."""
        L = self.get_L()
        return L.T @ L

    def forward(self, a, b):
        """
        Compute distance d(a,b) between two amino acids.

        Args:
            a: amino acid code (str) or index (int)
            b: amino acid code (str) or index (int)

        Returns:
            scalar distance tensor
        """
        # Convert indices to codes if needed
        if isinstance(a, int):
            a = self._aa_list[a]
        if isinstance(b, int):
            b = self._aa_list[b]

        x_a = self.feature_space.get_features(a)
        x_b = self.feature_space.get_features(b)

        delta = x_a - x_b
        M = self.get_M()
        dist_sq = delta @ M @ delta
        distance = torch.tanh(dist_sq)
        return distance

    def compute_distance_matrix(self):
        """
        Compute the full 20x20 distance matrix.

        Returns:
            torch.Tensor of shape (20, 20)
        """
        n = len(self._aa_list)
        D = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                D[i, j] = self.forward(self._aa_list[i], self._aa_list[j])
        return D

    def compute_distance_sq(self, a, b):
        """Compute squared Mahalanobis distance (before tanh)."""
        if isinstance(a, int):
            a = self._aa_list[a]
        if isinstance(b, int):
            b = self._aa_list[b]

        x_a = self.feature_space.get_features(a)
        x_b = self.feature_space.get_features(b)
        delta = x_a - x_b
        M = self.get_M()
        return delta @ M @ delta

    def asymmetry_penalty(self):
        """
        Penalty for asymmetry: should be zero since our metric is symmetric by construction.
        Included for completeness and analysis.
        """
        n = len(self._aa_list)
        penalty = 0.0
        for i in range(n):
            for j in range(n):
                d_ij = self.forward(self._aa_list[i], self._aa_list[j])
                d_ji = self.forward(self._aa_list[j], self._aa_list[i])
                penalty += (d_ij - d_ji) ** 2
        return penalty / (n * n)

    def forward_all_pairs(self):
        """
        Compute distances for all 20x20 pairs efficiently.
        Returns tensor of shape (20, 20).
        """
        X = self.feature_space.get_all_features()
        M = self.get_M()
        # Compute pairwise squared Mahalanobis distances
        # D[i,j] = (x_i - x_j)^T M (x_i - x_j)
        XM = X @ M  # (20, d)
        XM_XT = XM @ X.T  # (20, 20)
        diag = torch.diag(XM_XT)  # (20,)
        D_sq = diag.unsqueeze(0) + diag.unsqueeze(1) - 2 * XM_XT
        D_sq = torch.clamp(D_sq, min=0.0)
        return torch.tanh(D_sq)