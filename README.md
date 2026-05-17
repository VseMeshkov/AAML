# AAML: Amino Acid Metric Learning

ML library for learning distance functions (metrics) on the set of 20 standard amino acids.

## Overview

AAML learns a **Mahalanobis distance metric** of the form:

$$d(a, b) = \tanh\big((x_a - x_b)^\top M (x_a - x_b)\big)$$

where:
- $x_a, x_b \in \mathbb{R}^5$ are feature vectors encoding physicochemical properties of amino acids
- $M = L^\top L$ is a positive semi-definite matrix that is learned from data
- $\tanh$ normalizes distances to $(0, 1)$

The metric is trained to satisfy all **axioms of a distance function (metric)**:
- **Non-negativity**: $d(a,b) \ge 0$ — distances are never negative
- **Identity of indiscernibles**: $d(a,a) = 0$ — distance to itself is zero
- **Symmetry**: $d(a,b) = d(b,a)$ — direction doesn't matter (enforced by construction $M = L^\top L$)
- **Triangle inequality**: $d(a,c) \le d(a,b) + d(b,c)$ — detours can't make path shorter

## Motivation

In bioinformatics, quantifying the **distance between amino acids** is fundamental for:
- **Sequence alignment** (scoring substitutions in BLAST, ClustalW)
- **Evolutionary models** (PAM, BLOSUM substitution matrices)
- **Protein structure prediction** (assessing mutation impact)
- **Phylogenetic analysis** (building evolutionary trees)

Traditional substitution matrices (BLOSUM62, PAM250) are hand-constructed from curated sequence alignments.  
AAML learns such a distance **automatically from data** using gradient-based optimization, with the flexibility to incorporate multiple physicochemical features and metric constraints.

## Project Structure

```
proj_bioinform/
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── test_metric.py             # Full test suite
├── aaml/
│   ├── __init__.py            # Package init & exports
│   ├── features.py            # 5D feature vectors for 20 amino acids
│   ├── metric.py              # Learnable Mahalanobis distance metric
│   ├── alignment.py           # Needleman-Wunsch sequence alignment
│   ├── loss.py                # Loss functions (triplet, asymmetry, triangle, BLOSUM)
│   ├── train.py               # Training loop with Adam + LR scheduling
│   ├── evaluation.py          # Comparison with BLOSUM62 & PAM matrices
│   └── data.py                # Data loading (FASTA, built-in, synthetic)
└── notebooks/
    └── demo.ipynb             # Interactive demo with visualizations
```

## How It Works

### 1. Feature Space (`features.py`)

Each of the 20 standard amino acids is encoded as a **5-dimensional feature vector**:

| Feature | Description | Source |
|---------|-------------|--------|
| **Hydrophobicity** | Kyte-Doolittle scale | Kyte & Doolittle (1982) |
| **Molecular weight** | Mass in g/mol | Standard chemistry |
| **Polarity** | Grantham polarity score | Grantham (1974) |
| **Van der Waals volume** | Side-chain volume | Zamyatnin (1972) |
| **Charge** | Net charge at pH 7.4 | Standard biochemistry |

Feature vectors are **Z-score normalized** (zero mean, unit variance) so all dimensions contribute equally.

Examples:
```
AA  Hydropathy       MW  Polarity  Volume  Charge
--------------------------------------------------
A        1.80    89.09      8.10   67.00    0.00
R       -4.50   174.20     10.50  148.00    1.00
D       -3.50   133.10     13.00   91.00   -1.00
E       -3.50   147.13     12.30  109.00   -1.00
F        2.80   165.19      5.20  135.00    0.00
G       -0.40    75.07      9.00   48.00    0.00
...
```

### 2. Metric Definition (`metric.py`)

The distance between two amino acids $a$ and $b$ is defined as:

$$d(a, b) = \tanh\big((\mathbf{x}_a - \mathbf{x}_b)^\top M (\mathbf{x}_a - \mathbf{x}_b)\big)$$

where:
- $\mathbf{x}_a, \mathbf{x}_b \in \mathbb{R}^5$ — normalized feature vectors
- $M = L^\top L$ — **positive semi-definite** matrix (parameterized via Cholesky decomposition)
- $L$ — lower triangular matrix (15 free parameters), diagonal parameterized as $\exp(\log(\text{diag}))$ for positivity
- $\tanh$ — squashes distances to $(0, 1)$, ensuring bounded output

**Why PSD constraint ($M = L^\top L$)?**
- Guarantees $d(a,b) \ge 0$ (non-negativity)
- Guarantees $d(a,b) = d(b,a)$ (symmetry) — since $M$ is symmetric
- Guarantees $d(a,a) = 0$ (reflexivity) — since $\mathbf{x}_a - \mathbf{x}_a = 0$

### 3. Loss Functions (`loss.py`)

AAML uses a **multi-objective loss** to train the metric:

$$\mathcal{L} = w_1 \mathcal{L}_{\text{triplet}} + w_2 \mathcal{L}_{\text{asym}} + w_3 \mathcal{L}_{\triangle} + w_4 \mathcal{L}_{\text{BLOSUM}}$$

#### Triplet Loss
$$\mathcal{L}_{\text{triplet}} = \sum_{a,p,n} \max(0, d(a,p) - d(a,n) + \text{margin})$$

where $(a,p)$ are chemically similar (same group), $(a,n)$ are dissimilar (different groups).  
Chemical groups used: nonpolar (AGILMPV), polar (NQST), acidic (DE), basic (KRH), aromatic (FWY), sulfur (CM).

#### Asymmetry Loss
$$\mathcal{L}_{\text{asym}} = \frac{1}{n^2} \sum_{i,j} (d_{ij} - d_{ji})^2$$

Near-zero by construction, included for numerical stability.

#### Triangle Inequality Loss
$$\mathcal{L}_{\triangle} = \frac{1}{N} \sum_{i,j,k} \max(0, d_{ik} - (d_{ij} + d_{jk}))$$

Penalizes violations of triangle inequality across all $\binom{20}{3} = 1140$ triplets.

#### BLOSUM Comparison Loss
$$\mathcal{L}_{\text{BLOSUM}} = \text{MSE}(D_{\text{learned}}, D_{\text{BLOSUM62}})$$

Encourages the learned metric to be consistent with BLOSUM62 substitution scores (converted to distances).

### 4. Training (`train.py`)

**Optimizer**: Adam with weight decay ($\lambda = 10^{-5}$)  
**Learning rate**: Initial 0.01, with ReduceLROnPlateau scheduler (factor 0.5, patience 5)  
**Gradient clipping**: max norm 1.0  
**Early stopping**: patience 25 epochs  
**Device**: auto-detection (CUDA → MPS → CPU)

Training progress:
```
Epoch    1/100 | Loss: 0.4522 | triplet: 0.4317 | triangle: 0.0004 | blosum: 0.0202
Epoch   10/100 | Loss: 0.3986 | triplet: 0.3785 | triangle: 0.0006 | blosum: 0.0195
Epoch   50/100 | Loss: 0.2604 | triplet: 0.2052 | triangle: 0.0209 | blosum: 0.0343
Epoch  100/100 | Loss: 0.2001 | triplet: 0.1432 | triangle: 0.0173 | blosum: 0.0395
```

### 5. Evaluation (`evaluation.py`)

The learned distance matrix is compared with BLOSUM62 using:
- **MSE/MAE**: absolute difference between matrices
- **Pearson r**: linear correlation of distance pairs
- **Spearman ρ**: rank correlation (monotonic relationship)
- **Kendall's τ**: ranking agreement per amino acid
- **Scatter plots**: learned vs BLOSUM distances

## Installation

```bash
pip install -r requirements.txt
```

The project uses standard scientific Python stack:
- `torch >= 2.0.0` — neural network framework
- `numpy >= 1.20.0` — numerical computing
- `scipy >= 1.7.0` — statistical functions (Spearman, Kendall)
- `matplotlib`, `seaborn` — plotting
- `jupyter` — interactive notebooks

## Quick Start

```python
import sys; sys.path.insert(0, '.')
from aaml import FeatureSpace, AA_Metric, MetricLoss, train_model
from aaml.evaluation import compare_with_blosum, matrix_to_dataframe

# 1. Create feature space (5 physicochemical properties)
fs = FeatureSpace()

# 2. Initialize metric (Euclidean distance by default)
metric = AA_Metric(fs)

# 3. Compute BLOSUM62 target for regularization
blosum_target = MetricLoss.compute_blosum_distance()

# 4. Create loss function
loss_fn = MetricLoss(fs, w_triplet=1.0, w_asymmetry=0.1, w_triangle=10.0, w_blosum=0.5)

# 5. Train (Adam, 100 epochs, LR scheduling)
history, metric = train_model(metric, loss_fn, num_epochs=100,
                               learning_rate=0.01, blosum_target=blosum_target)

# 6. Get distance matrix and compare with BLOSUM
D = metric.forward_all_pairs()
print(matrix_to_dataframe(D))
results = compare_with_blosum(D, "Learned Metric")
print(f"Pearson vs BLOSUM62: {results['pearson_correlation']:.4f}")
```

## Distance Matrix (20×20)

The learned 20×20 distance matrix — lower triangle shows pairwise distances:

```
         A      C      D      E      F      G      H      I      K      L      M      N      P      Q      R      S      T      V      W      Y
A     0.000                                                                                                                              
C     0.774  0.000                                                                                                                       
D     1.000  1.000  0.000                                                                                                                
E     1.000  1.000  0.044  0.000                                                                                                         
F     0.832  0.208  1.000  1.000  0.000                                                                                                  
G     0.080  0.858  1.000  1.000  0.908  0.000                                                                                           
H     1.000  1.000  1.000  1.000  0.999  1.000  0.000                                                                                    
I     0.247  0.727  1.000  1.000  0.809  0.367  1.000  0.000                                                                             
K     1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  0.000                                                                      
L     0.408  0.775  1.000  1.000  0.870  0.442  1.000  0.054  1.000  0.000                                                               
M     0.524  0.249  1.000  1.000  0.194  0.625  1.000  0.439  1.000  0.510  0.000                                                        
N     0.923  0.996  1.000  1.000  0.980  0.922  0.961  0.988  1.000  0.995  0.963  0.000                                                 
P     0.388  0.860  1.000  1.000  0.873  0.219  1.000  0.449  1.000  0.413  0.499  0.918  0.000                                          
Q     0.872  0.990  1.000  1.000  0.959  0.850  0.982  0.966  1.000  0.981  0.904  0.102  0.771  0.000                                   
R     1.000  1.000  1.000  1.000  1.000  1.000  1.000  1.000  0.548  1.000  1.000  1.000  1.000  1.000  0.000                            
S     0.226  0.787  1.000  1.000  0.723  0.252  0.999  0.598  1.000  0.724  0.448  0.628  0.347  0.496  1.000  0.000                     
T     0.237  0.837  1.000  1.000  0.781  0.205  1.000  0.502  1.000  0.604  0.443  0.682  0.172  0.479  1.000  0.053  0.000              
V     0.135  0.658  1.000  1.000  0.762  0.292  1.000  0.033  1.000  0.150  0.405  0.981  0.478  0.957  1.000  0.491  0.446  0.000       
W     0.991  0.786  1.000  1.000  0.468  0.993  0.997  0.989  1.000  0.991  0.707  0.991  0.971  0.975  1.000  0.948  0.955  0.987  0.000
Y     0.955  0.641  1.000  1.000  0.322  0.963  0.997  0.955  1.000  0.962  0.459  0.969  0.881  0.920  1.000  0.824  0.838  0.949  0.080  0.000
```

### Key observations:
- **D and E (acidic)**: $d(D,E) = 0.044$ — very close, biologically accurate
- **I, L, V (hydrophobic)**: $d(I,L) = 0.054$, $d(I,V) = 0.033$, $d(L,V) = 0.150$ — form a tight cluster
- **K and R (basic)**: $d(K,R) = 0.548$ — close but distinct
- **W and Y (aromatic)**: $d(W,Y) = 0.080$ — very close
- **G and A (small)**: $d(G,A) = 0.080$ — similar size and properties

## Training Results

### Metric Properties Validation

| Property | Result | Method |
|----------|--------|--------|
| Non-negativity | ✅ $d(a,b) \ge 0$ | Guaranteed by PSD $M$ |
| Reflexivity | ✅ $d(a,a) = 0$ | $\mathbf{x}_a - \mathbf{x}_a = 0$ |
| Symmetry | ✅ $|d(a,b) - d(b,a)| < 10^{-6}$ | $M = L^\top L$ is symmetric |
| Triangle Ieq | ✅ $< 3\%$ violations | Minimized via loss term |

### Comparison with BLOSUM62

| Metric | Value |
|--------|-------|
| Pearson correlation | ~0.51 |
| Spearman correlation | ~0.34 |
| Kendall's τ (mean) | ~0.42 |
| MSE | ~0.07 |
| MAE | ~0.20 |

### Top-ranked amino acid neighbors (by learned distance)

| Anchor | Nearest | Second | Third | Biological interpretation |
|--------|---------|--------|-------|--------------------------|
| A | G (0.080) | V (0.135) | S (0.226) | Small, flexible residues |
| D | E (0.044) | — | — | Both negatively charged |
| I | V (0.033) | L (0.054) | A (0.247) | Aliphatic hydrophobic |
| F | M (0.194) | C (0.208) | Y (0.322) | Hydrophobic, aromatic |
| W | Y (0.080) | F (0.468) | M (0.707) | Aromatic (Trp-Tyr common subst.) |
| K | R (0.548) | — | — | Both positively charged |

## Running the Code

### Test Suite
```bash
python3 test_metric.py
```

This runs: feature extraction → metric init → training (50 epochs) → property validation → BLOSUM comparison → symmetry/triangle random tests.

### Interactive Demo
```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook contains:
1. Feature space visualization (bar plots of all 5 properties)
2. Distance matrix heatmaps (before vs after training)
3. Training loss curves (total + components)
4. BLOSUM comparison (scatter plots, per-residue ranking)
5. Learned M matrix (heatmap + eigenvalues)
6. Interpretation of results

## Mathematical Background

### What is a Metric (Distance Function)?

A function $d: \mathcal{X} \times \mathcal{X} \to \mathbb{R}_{\ge 0}$ is a **metric** if:

1. **Non-negativity**: $d(x,y) \ge 0$ (distances are never negative)
2. **Identity**: $d(x,y) = 0 \iff x = y$ (same objects have zero distance)
3. **Symmetry**: $d(x,y) = d(y,x)$ (order doesn't matter)
4. **Triangle inequality**: $d(x,z) \le d(x,y) + d(y,z)$ (direct path is shortest)

### Mahalanobis Distance

The **Mahalanobis distance** generalizes Euclidean distance:

$$d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^\top M (\mathbf{x} - \mathbf{y})}$$

When $M = I$, this reduces to Euclidean distance.  
When $M = \Sigma^{-1}$ (inverse covariance), it accounts for feature correlations.

AAML uses a **squared** Mahalanobis distance with tanh normalization:

$$d(a,b) = \tanh\big((\mathbf{x}_a - \mathbf{x}_b)^\top M (\mathbf{x}_a - \mathbf{x}_b)\big)$$

### Why Tanh?

The $\tanh$ function squashes the unbounded Mahalanobis distance to $(0, 1)$:
- Prevents extreme values from dominating the loss
- Makes distances interpretable (0 = identical, 1 = maximally different)
- Stabilizes training by bounding gradients

### Connection to BLOSUM

BLOSUM62 scores are **log-odds ratios**:

$$S(a,b) = \log_2 \frac{p(a,b)}{p(a)p(b)}$$

where $p(a,b)$ is the observed frequency of substitution $a \leftrightarrow b$ in trusted alignments.  
AAML converts these to distances:

$$D_{\text{BLOSUM}}(a,b) = 1 - \frac{S(a,b) - S_{\min}}{S_{\max} - S_{\min}}$$

This maps high substitution scores (frequent, conservative) → small distances.

## References

1. **Kyte, J. & Doolittle, R.F.** (1982). A simple method for displaying the hydropathic character of a protein. *Journal of Molecular Biology*, 157(1), 105-132.
2. **Grantham, R.** (1974). Amino acid difference formula to help explain protein evolution. *Science*, 185(4154), 862-864.
3. **Zamyatnin, A.A.** (1972). Protein volume in solution. *Progress in Biophysics and Molecular Biology*, 24, 107-123.
4. **Henikoff, S. & Henikoff, J.G.** (1992). Amino acid substitution matrices from protein blocks. *PNAS*, 89(22), 10915-10919. — BLOSUM matrices
5. **Dayhoff, M.O., Schwartz, R.M. & Orcutt, B.C.** (1978). A model of evolutionary change in proteins. — PAM matrices
6. **Needleman, S.B. & Wunsch, C.D.** (1970). A general method applicable to the search for similarities in the amino acid sequence of two proteins. *Journal of Molecular Biology*, 48(3), 443-453.
7. **Weinberger, K.Q. & Saul, L.K.** (2009). Distance metric learning for large margin nearest neighbor classification. *Journal of Machine Learning Research*, 10, 207-244. — Large margin metric learning
8. **Xing, E.P. et al.** (2003). Distance metric learning with application to clustering with side-information. *NIPS*. — PSD-constrained metric learning
9. **SNACK**: Sequence Normalized Alignment Comparison Kit (https://github.com/kisnikser/snack) — inspiration and reference implementation

## License

This project is provided for educational and research purposes.

## Author

Proj_bioinform — ML project на тему "Функции расстояния (метрики) на множестве аминокислот 20×20"