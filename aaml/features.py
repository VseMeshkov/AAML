"""
Feature space for 20 standard amino acids.

Each amino acid is encoded as a 5-dimensional feature vector:
1. Hydrophobicity (Kyte-Doolittle scale)
2. Molecular weight (normalized)
3. Polarity
4. van der Waals volume
5. Charge at pH 7.4
"""

import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class FeatureSpace:
    """Feature space encoding 20 standard amino acids into 5D vectors."""

    def __init__(self, device=None):
        self.feature_dim = 5
        self.device = device
        self._cache = {}

        # 1. Hydrophobicity (Kyte-Doolittle scale)
        self.hydropathy = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }

        # 2. Molecular weight
        self.mw = {
            'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
            'G': 75.07, 'H': 155.16, 'I': 131.18, 'K': 146.19, 'L': 131.18,
            'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
            'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
        }

        # 3. Polarity (Grantham, 1974)
        self.polarity = {
            'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.3, 'F': 5.2,
            'G': 9.0, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9,
            'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5,
            'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2
        }

        # 4. van der Waals volume (Zamyatnin, 1972)
        self.volume = {
            'A': 67, 'C': 86, 'D': 91, 'E': 109, 'F': 135,
            'G': 48, 'H': 118, 'I': 124, 'K': 135, 'L': 124,
            'M': 124, 'N': 96, 'P': 90, 'Q': 114, 'R': 148,
            'S': 73, 'T': 93, 'V': 105, 'W': 163, 'Y': 141
        }

        # 5. Charge at pH 7.4
        self.charge = {
            'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
            'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
            'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        }

        # Stats for normalization (computed from the data)
        self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute mean and std for each feature to normalize."""
        keys = list(self.hydropathy.keys())
        props = ['hydropathy', 'mw', 'polarity', 'volume', 'charge']
        self.norm_mean = {}
        self.norm_std = {}

        for prop in props:
            values = [getattr(self, prop)[k] for k in keys]
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            self.norm_mean[prop] = mean
            self.norm_std[prop] = std if std > 1e-8 else 1.0

    def get_raw_features(self, aa):
        """Return raw (unnormalized) 5D feature vector for an amino acid."""
        return torch.tensor([
            self.hydropathy[aa],
            self.mw[aa],
            self.polarity[aa],
            self.volume[aa],
            self.charge[aa]
        ], dtype=torch.float32)

    def get_features(self, aa):
        """Return normalized 5D feature vector for an amino acid."""
        if aa in self._cache:
            return self._cache[aa]

        raw = torch.tensor([
            (self.hydropathy[aa] - self.norm_mean['hydropathy']) / self.norm_std['hydropathy'],
            (self.mw[aa] - self.norm_mean['mw']) / self.norm_std['mw'],
            (self.polarity[aa] - self.norm_mean['polarity']) / self.norm_std['polarity'],
            (self.volume[aa] - self.norm_mean['volume']) / self.norm_std['volume'],
            (self.charge[aa] - self.norm_mean['charge']) / self.norm_std['charge'],
        ], dtype=torch.float32)

        if self.device is not None:
            raw = raw.to(self.device)

        self._cache[aa] = raw
        return raw

    def get_all_features(self):
        """Return feature matrix of shape (20, 5) for all amino acids."""
        features = []
        for aa in AMINO_ACIDS:
            features.append(self.get_features(aa))
        return torch.stack(features)

    def get_amino_acid_list(self):
        """Return list of standard amino acid codes."""
        return list(AMINO_ACIDS)

    def to(self, device):
        """Move feature space to a device, clear cache."""
        self.device = device
        self._cache = {k: v.to(device) for k, v in self._cache.items()}
        return self