#!/usr/bin/env python3
"""
Quick test script for AAML: Amino Acid Metric Learning.

Trains a distance metric on the 20 amino acids and evaluates
its properties (symmetry, triangle inequality, comparison with BLOSUM62).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np


def main():
    print("=" * 70)
    print("AAML: Amino Acid Metric Learning — Test Suite")
    print("=" * 70)

    # 1. Feature space
    print("\n[1] Creating FeatureSpace...")
    from aaml import FeatureSpace
    fs = FeatureSpace()
    print(f"    Feature dimension: {fs.feature_dim}")
    print(f"    Feature vector for A: {fs.get_features('A')}")
    print(f"    Feature vector for R: {fs.get_features('R')}")

    # Print feature table
    print(f"\n    {'AA':>3} {'Hydropathy':>10} {'MW':>8} {'Polarity':>8} {'Volume':>8} {'Charge':>8}")
    print(f"    {'-'*42}")
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        raw = fs.get_raw_features(aa)
        print(f"    {aa:>3} {raw[0]:10.2f} {raw[1]:8.2f} {raw[2]:8.2f} {raw[3]:8.2f} {raw[4]:8.2f}")

    # 2. Metric initialization
    print("\n[2] Creating AA_Metric (init = Euclidean)...")
    from aaml import AA_Metric
    metric = AA_Metric(fs)
    print(f"    d(A, A) = {metric('A', 'A'):.6f}  (should be ~0)")
    print(f"    d(A, R) = {metric('A', 'R'):.6f}")
    print(f"    d(R, A) = {metric('R', 'A'):.6f}  (should equal d(A,R))")
    print(f"    d(A, V) = {metric('A', 'V'):.6f}")
    print(f"    d(A, C) = {metric('A', 'C'):.6f}")

    # Initial distance matrix
    print("\n[3] Initial distance matrix (Euclidean):")
    D_init = metric.forward_all_pairs()
    from aaml.evaluation import matrix_to_dataframe
    print(matrix_to_dataframe(D_init))

    # 4. Training
    print("\n[4] Training the metric...")
    from aaml.loss import MetricLoss
    from aaml.train import train_model, compute_metric_properties

    blosum_target = MetricLoss.compute_blosum_distance()

    loss_fn = MetricLoss(
        fs,
        w_triplet=1.0,
        w_asymmetry=0.1,
        w_triangle=10.0,
        w_blosum=0.5
    )

    history, metric_trained = train_model(
        metric, loss_fn,
        num_epochs=200,
        learning_rate=0.01,
        blosum_target=blosum_target,
        verbose=True,
        early_stopping_patience=30
    )

    # 5. Results
    print("\n[5] Learned distance matrix:")
    D_learned = metric_trained.forward_all_pairs()
    print(matrix_to_dataframe(D_learned))

    # 6. Metric properties analysis
    print("\n[6] Metric properties analysis:")
    props = compute_metric_properties(metric_trained)
    print(f"    Non-negative:          {props['non_negative']}")
    print(f"    Reflexive (d(a,a)=0):  {props['reflexive']}")
    print(f"    Max asymmetry error:   {props['asymmetry_max']:.2e}")
    print(f"    Triangle violations:   {props['triangle_violations']} "
          f"({props['triangle_violation_pct']:.4f}%)")
    print(f"    Max violation:         {props['max_triangle_violation']:.2e}")
    print(f"    IS VALID METRIC:       {props['is_metric']}")

    # 7. Comparison with BLOSUM62
    print("\n[7] Comparison with BLOSUM62:")
    from aaml.evaluation import compare_with_blosum, compute_ranking_correlation

    results_blosum = compare_with_blosum(D_learned, "Learned Metric")
    print(f"    MSE:                  {results_blosum['mse']:.6f}")
    print(f"    MAE:                  {results_blosum['mae']:.6f}")
    print(f"    Pearson correlation:  {results_blosum['pearson_correlation']:.4f}")
    print(f"    Spearman correlation: {results_blosum['spearman_correlation']:.4f}")

    tau_values, mean_tau = compute_ranking_correlation(D_learned, results_blosum['blosum_distance'])
    print(f"    Kendall's tau (mean): {mean_tau:.4f}")
    print(f"    Tau per amino acid:")
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        print(f"      {aa}: {tau_values[aa]:.4f}")

    # 8. Training history summary
    print("\n[8] Training history:")
    print(f"    Epochs completed: {len(history['epoch'])}")
    print(f"    Initial loss:     {history['total_loss'][0]:.6f}")
    print(f"    Final loss:       {history['total_loss'][-1]:.6f}")
    print(f"    Best loss:        {min(history['total_loss']):.6f}")

    # 9. Symmetry test
    print("\n[9] Explicit symmetry test (10 random pairs):")
    import random
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    max_sym_err = 0.0
    for _ in range(10):
        a, b = random.sample(aa_list, 2)
        d1 = metric_trained(a, b).item()
        d2 = metric_trained(b, a).item()
        err = abs(d1 - d2)
        max_sym_err = max(max_sym_err, err)
        print(f"    d({a},{b})={d1:.6f}  d({b},{a})={d2:.6f}  diff={err:.2e}")
    print(f"    Max symmetry error: {max_sym_err:.2e}")

    # 10. Triangle inequality test (random triplets)
    print("\n[10] Triangle inequality test (10 random triplets):")
    max_tri_viol = 0.0
    viol_count = 0
    for _ in range(10):
        a, b, c = random.sample(aa_list, 3)
        dab = metric_trained(a, b).item()
        dbc = metric_trained(b, c).item()
        dac = metric_trained(a, c).item()
        viol = max(0.0, dac - (dab + dbc))
        if viol > 1e-6:
            viol_count += 1
            max_tri_viol = max(max_tri_viol, viol)
        status = "VIOLATED" if viol > 1e-6 else "OK"
        print(f"    d({a},{c})={dac:.4f} ≤ d({a},{b})+d({b},{c})={dab+dbc:.4f}  [{status}]")
    print(f"    Violations: {viol_count}/10, Max violation: {max_tri_viol:.2e}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    return props['is_metric']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)