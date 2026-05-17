"""
Loss functions for training the amino acid distance metric.

Includes:
1. Triplet loss: enforce that chemically similar AAs are closer
2. Asymmetry loss: penalty for asymmetry (d(a,b) != d(b,a))
3. Triangle inequality loss: penalty for violations
4. Alignment loss: from sequence alignment quality
5. BLOSUM comparison loss: match known substitution matrices
"""

import torch


class MetricLoss(torch.nn.Module):
    """
    Combined loss for training AA_Metric.

    L = w1 * L_triplet + w2 * L_asymmetry + w3 * L_triangle + w4 * L_blosum
    """

    def __init__(self, feature_space, w_triplet=1.0, w_asymmetry=0.1,
                 w_triangle=10.0, w_blosum=0.5, margin=0.5):
        super().__init__()
        self.feature_space = feature_space
        self.aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        self.w_triplet = w_triplet
        self.w_asymmetry = w_asymmetry
        self.w_triangle = w_triangle
        self.w_blosum = w_blosum
        self.margin = margin

    def forward(self, metric, blosum_target=None):
        """
        Compute total loss.

        Args:
            metric: AA_Metric instance
            blosum_target: optional 20x20 target distance matrix from BLOSUM

        Returns:
            total_loss, loss_components dict
        """
        # Compute full distance matrix
        D = metric.forward_all_pairs()

        losses = {}

        if self.w_triplet > 0:
            losses['triplet'] = self.w_triplet * self._triplet_loss(D)

        if self.w_asymmetry > 0:
            losses['asymmetry'] = self.w_asymmetry * self._asymmetry_loss(D)

        if self.w_triangle > 0:
            losses['triangle'] = self.w_triangle * self._triangle_inequality_loss(D)

        if self.w_blosum > 0 and blosum_target is not None:
            losses['blosum'] = self.w_blosum * self._blosum_comparison_loss(D, blosum_target)

        total = sum(losses.values()) if losses else torch.tensor(0.0, device=D.device)
        return total, losses

    def _triplet_loss(self, D):
        """
        Triplet loss to encourage meaningful distances.

        Uses chemical grouping: AAs in same group should be closer
        than AAs in different groups.
        """
        # Define chemical groups
        groups = {
            'nonpolar': ['A', 'G', 'I', 'L', 'M', 'P', 'V'],
            'polar': ['N', 'Q', 'S', 'T'],
            'acidic': ['D', 'E'],
            'basic': ['R', 'H', 'K'],
            'aromatic': ['F', 'W', 'Y'],
            'sulfur': ['C', 'M']
        }

        idx_map = {aa: i for i, aa in enumerate(self.aa_list)}
        loss = 0.0
        count = 0

        for group_name, members in groups.items():
            members_idx = [idx_map[m] for m in members if m in idx_map]
            others_idx = [idx_map[m] for m in self.aa_list
                          if m not in members and m in idx_map]

            for anchor in members_idx:
                for pos in members_idx:
                    if pos == anchor:
                        continue
                    for neg in others_idx:
                        d_pos = D[anchor, pos]
                        d_neg = D[anchor, neg]
                        loss += torch.clamp(d_pos - d_neg + self.margin, min=0.0)
                        count += 1

        return loss / max(count, 1)

    def _asymmetry_loss(self, D):
        """Penalize asymmetry: sum (d_ij - d_ji)^2."""
        loss = torch.sum((D - D.T) ** 2)
        n = D.shape[0]
        return loss / (n * n)

    def _triangle_inequality_loss(self, D):
        """
        Penalize violations of triangle inequality:
        d(i,k) <= d(i,j) + d(j,k) for all i,j,k
        """
        n = D.shape[0]
        loss = 0.0
        count = 0

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i == j or j == k or i == k:
                        continue
                    violation = D[i, k] - (D[i, j] + D[j, k])
                    loss += torch.clamp(violation, min=0.0)
                    count += 1

        return loss / max(count, 1)

    def _blosum_comparison_loss(self, D, target):
        """
        MSE loss between learned distance matrix and BLOSUM-derived target.

        The target should be a distance matrix derived from BLOSUM scores.
        """
        return torch.nn.functional.mse_loss(D, target)

    @staticmethod
    def compute_blosum_distance():
        """
        Compute a distance matrix from BLOSUM62 substitution scores.

        Converts BLOSUM62 scores to distances: d = 1 - (S - min)/(max - min)
        """
        try:
            from Bio.SubsMat import MatrixInfo
            blosum = MatrixInfo.blosum62
        except ImportError:
            # Fallback: use a simplified BLOSUM62
            blosum = _get_default_blosum62()

        # Extract all pairwise scores
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        n = len(aa_list)
        S = torch.zeros(n, n)
        idx_map = {aa: i for i, aa in enumerate(aa_list)}

        for (a, b), score in blosum.items():
            if a in idx_map and b in idx_map:
                i, j = idx_map[a], idx_map[b]
                S[i, j] = score
                S[j, i] = score

        # Fill diagonal with self-scores
        for aa, i in idx_map.items():
            S[i, i] = blosum.get((aa, aa), 4)

        # Convert scores to distances
        min_s, max_s = S.min(), S.max()
        D = 1 - (S - min_s) / (max_s - min_s)
        return D


def _get_default_blosum62():
    """Simplified BLOSUM62 for when BioPython is not available."""
    return {
        ('A', 'A'): 4, ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2,
        ('A', 'C'): 0, ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0,
        ('A', 'H'): -2, ('A', 'I'): -1, ('A', 'L'): -1, ('A', 'K'): -1,
        ('A', 'M'): -1, ('A', 'F'): -2, ('A', 'P'): -1, ('A', 'S'): 1,
        ('A', 'T'): 0, ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,
        ('R', 'R'): 5, ('R', 'N'): 0, ('R', 'D'): -2, ('R', 'C'): -3,
        ('R', 'Q'): 1, ('R', 'E'): 0, ('R', 'G'): -2, ('R', 'H'): 0,
        ('R', 'I'): -3, ('R', 'L'): -2, ('R', 'K'): 2, ('R', 'M'): -1,
        ('R', 'F'): -3, ('R', 'P'): -2, ('R', 'S'): -1, ('R', 'T'): -1,
        ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -3,
        ('N', 'N'): 6, ('N', 'D'): 1, ('N', 'C'): -3, ('N', 'Q'): 0,
        ('N', 'E'): 0, ('N', 'G'): 1, ('N', 'H'): 1, ('N', 'I'): -3,
        ('N', 'L'): -3, ('N', 'K'): 0, ('N', 'M'): -2, ('N', 'F'): -3,
        ('N', 'P'): -2, ('N', 'S'): 1, ('N', 'T'): 0, ('N', 'W'): -4,
        ('N', 'Y'): -2, ('N', 'V'): -3,
        ('D', 'D'): 6, ('D', 'C'): -3, ('D', 'Q'): 0, ('D', 'E'): 2,
        ('D', 'G'): -1, ('D', 'H'): -1, ('D', 'I'): -3, ('D', 'L'): -4,
        ('D', 'K'): -1, ('D', 'M'): -3, ('D', 'F'): -3, ('D', 'P'): -1,
        ('D', 'S'): 0, ('D', 'T'): -1, ('D', 'W'): -4, ('D', 'Y'): -3,
        ('D', 'V'): -3,
        ('C', 'C'): 9, ('C', 'Q'): -3, ('C', 'E'): -4, ('C', 'G'): -3,
        ('C', 'H'): -3, ('C', 'I'): -1, ('C', 'L'): -1, ('C', 'K'): -3,
        ('C', 'M'): -1, ('C', 'F'): -2, ('C', 'P'): -3, ('C', 'S'): -1,
        ('C', 'T'): -1, ('C', 'W'): -2, ('C', 'Y'): -2, ('C', 'V'): -1,
        ('Q', 'Q'): 5, ('Q', 'E'): 2, ('Q', 'G'): -2, ('Q', 'H'): 0,
        ('Q', 'I'): -3, ('Q', 'L'): -2, ('Q', 'K'): 1, ('Q', 'M'): 0,
        ('Q', 'F'): -3, ('Q', 'P'): -1, ('Q', 'S'): 0, ('Q', 'T'): -1,
        ('Q', 'W'): -2, ('Q', 'Y'): -1, ('Q', 'V'): -2,
        ('E', 'E'): 5, ('E', 'G'): -2, ('E', 'H'): 0, ('E', 'I'): -3,
        ('E', 'L'): -3, ('E', 'K'): 1, ('E', 'M'): -2, ('E', 'F'): -3,
        ('E', 'P'): -1, ('E', 'S'): 0, ('E', 'T'): -1, ('E', 'W'): -3,
        ('E', 'Y'): -2, ('E', 'V'): -2,
        ('G', 'G'): 6, ('G', 'H'): -2, ('G', 'I'): -4, ('G', 'L'): -4,
        ('G', 'K'): -2, ('G', 'M'): -3, ('G', 'F'): -3, ('G', 'P'): -2,
        ('G', 'S'): 0, ('G', 'T'): -2, ('G', 'W'): -2, ('G', 'Y'): -3,
        ('G', 'V'): -3,
        ('H', 'H'): 8, ('H', 'I'): -3, ('H', 'L'): -3, ('H', 'K'): -1,
        ('H', 'M'): -2, ('H', 'F'): -1, ('H', 'P'): -2, ('H', 'S'): -1,
        ('H', 'T'): -2, ('H', 'W'): -2, ('H', 'Y'): 2, ('H', 'V'): -3,
        ('I', 'I'): 4, ('I', 'L'): 2, ('I', 'K'): -3, ('I', 'M'): 1,
        ('I', 'F'): 0, ('I', 'P'): -3, ('I', 'S'): -2, ('I', 'T'): -1,
        ('I', 'W'): -3, ('I', 'Y'): -1, ('I', 'V'): 3,
        ('L', 'L'): 4, ('L', 'K'): -2, ('L', 'M'): 2, ('L', 'F'): 0,
        ('L', 'P'): -3, ('L', 'S'): -2, ('L', 'T'): -1, ('L', 'W'): -2,
        ('L', 'Y'): -1, ('L', 'V'): 1,
        ('K', 'K'): 5, ('K', 'M'): -1, ('K', 'F'): -3, ('K', 'P'): -1,
        ('K', 'S'): 0, ('K', 'T'): -1, ('K', 'W'): -3, ('K', 'Y'): -2,
        ('K', 'V'): -2,
        ('M', 'M'): 5, ('M', 'F'): 0, ('M', 'P'): -2, ('M', 'S'): -1,
        ('M', 'T'): -1, ('M', 'W'): -1, ('M', 'Y'): -1, ('M', 'V'): 1,
        ('F', 'F'): 6, ('F', 'P'): -4, ('F', 'S'): -2, ('F', 'T'): -2,
        ('F', 'W'): 1, ('F', 'Y'): 3, ('F', 'V'): -1,
        ('P', 'P'): 7, ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'W'): -4,
        ('P', 'Y'): -3, ('P', 'V'): -2,
        ('S', 'S'): 4, ('S', 'T'): 1, ('S', 'W'): -3, ('S', 'Y'): -2,
        ('S', 'V'): -2,
        ('T', 'T'): 5, ('T', 'W'): -2, ('T', 'Y'): -2, ('T', 'V'): 0,
        ('W', 'W'): 11, ('W', 'Y'): 2, ('W', 'V'): -3,
        ('Y', 'Y'): 7, ('Y', 'V'): -1,
        ('V', 'V'): 4,
    }