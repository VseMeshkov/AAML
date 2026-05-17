"""
Training loop for the amino acid distance metric.

Includes gradient descent with:
- Adam optimizer with weight decay
- Learning rate scheduling
- Loss tracking across epochs
- Early stopping option
"""

import torch
import torch.optim as optim
import time
import sys


def get_device():
    """Determine the best available device for PyTorch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_model(metric, loss_fn, num_epochs=100, learning_rate=0.01,
                blosum_target=None, device=None, verbose=True,
                early_stopping_patience=20):
    """
    Train the AA_Metric model.

    Args:
        metric: AA_Metric instance
        loss_fn: MetricLoss instance
        num_epochs: number of training epochs
        learning_rate: initial learning rate
        blosum_target: optional 20x20 target distance matrix from BLOSUM
        device: training device (auto-detected if None)
        verbose: print progress
        early_stopping_patience: stop if no improvement for this many epochs

    Returns:
        history: dict with 'epoch', 'total_loss', and component losses
        trained metric
    """
    if device is None:
        device = get_device()

    if verbose:
        print(f"Training on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in metric.parameters())}")

    # Move model to device
    metric = metric.to(device)
    metric.feature_space.to(device)

    if blosum_target is not None:
        blosum_target = blosum_target.to(device)

    # Optimizer
    optimizer = optim.Adam(metric.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Training history
    history = {
        'epoch': [],
        'total_loss': [],
        'triplet_loss': [],
        'asymmetry_loss': [],
        'triangle_loss': [],
        'blosum_loss': [],
        'lr': []
    }

    best_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    try:
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()

            # Compute loss
            total_loss, loss_components = loss_fn(metric, blosum_target)

            # Backprop
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(metric.parameters(), max_norm=1.0)
            optimizer.step()

            # Record history
            current_lr = optimizer.param_groups[0]['lr']
            history['epoch'].append(epoch)
            history['total_loss'].append(total_loss.item())
            history['triplet_loss'].append(loss_components.get('triplet', torch.tensor(0.0)).item())
            history['asymmetry_loss'].append(loss_components.get('asymmetry', torch.tensor(0.0)).item())
            history['triangle_loss'].append(loss_components.get('triangle', torch.tensor(0.0)).item())
            history['blosum_loss'].append(loss_components.get('blosum', torch.tensor(0.0)).item())
            history['lr'].append(current_lr)

            # Scheduler step
            scheduler.step(total_loss)

            # Early stopping check
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == 1):
                elapsed = time.time() - start_time
                comp_str = " | ".join(
                    f"{k.split('_')[0]}: {v:.4f}"
                    for k, v in loss_components.items()
                )
                print(f"Epoch {epoch:4d}/{num_epochs} | "
                      f"Loss: {total_loss.item():.6f} | "
                      f"{comp_str} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {elapsed:.1f}s")

            # Early stopping
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement for {early_stopping_patience} epochs)")
                break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}")

    total_time = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Final loss: {history['total_loss'][-1]:.6f}")

    return history, metric


def compute_metric_properties(metric):
    """
    Analyze the learned metric properties.

    Returns dict with:
    - distance_matrix: 20x20 distance matrix
    - asymmetry: max asymmetry error
    - triangle_violations: number of triangle inequality violations
    - is_metric: whether all metric properties hold
    """
    D = metric.forward_all_pairs()
    n = D.shape[0]

    # Check symmetry
    asym_max = (D - D.T).abs().max().item()

    # Check triangle inequality
    violations = 0
    max_violation = 0.0
    tol = 1e-4  # numerical tolerance
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i == j or j == k or i == k:
                    continue
                viol = D[i, k] - (D[i, j] + D[j, k])
                if viol > tol:
                    violations += 1
                    max_violation = max(max_violation, viol.item())

    total_triangles = n * (n - 1) * (n - 2)

    # Check non-negativity
    nonneg = (D >= -tol).all().item()

    # Check reflexivity (diagonal = 0)
    reflex = (D.diag().abs() < tol).all().item()

    return {
        'distance_matrix': D,
        'asymmetry_max': asym_max,
        'triangle_violations': violations,
        'triangle_violation_pct': 100.0 * violations / max(total_triangles, 1),
        'max_triangle_violation': max_violation,
        'non_negative': nonneg,
        'reflexive': reflex,
        'is_metric': nonneg and reflex and asym_max < 1e-6 and max_violation <= tol
    }
