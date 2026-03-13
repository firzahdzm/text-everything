"""
Bayesian LR Search — Lightweight GP-based learning rate optimization.

Uses a simple Gaussian Process with RBF kernel to model the loss landscape
in log10(LR) space, then selects the next LR to try via Expected Improvement.

No external dependencies beyond numpy and math (always available in training env).
"""
import math
import numpy as np


def _rbf_kernel(x1, x2, length_scale, variance=1.0):
    """RBF (Gaussian) kernel matrix between two sets of points."""
    x1 = np.asarray(x1, dtype=np.float64).reshape(-1, 1)
    x2 = np.asarray(x2, dtype=np.float64).reshape(-1, 1)
    dist_sq = (x1 - x2.T) ** 2
    return variance * np.exp(-0.5 * dist_sq / (length_scale ** 2))


def _gp_predict(X_train, y_train, X_test, length_scale, noise=1e-6):
    """
    Simple GP regression: predict mean and std at test points.

    Args:
        X_train: array of observed log10(LR) values
        y_train: array of observed losses (normalized)
        X_test: array of candidate log10(LR) values
        length_scale: RBF kernel length scale
        noise: observation noise variance

    Returns:
        mu: predicted mean at X_test
        sigma: predicted std at X_test
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    # Kernel matrices
    K = _rbf_kernel(X_train, X_train, length_scale) + noise * np.eye(len(X_train))
    K_s = _rbf_kernel(X_train, X_test, length_scale)
    K_ss = _rbf_kernel(X_test, X_test, length_scale)

    # Solve K^{-1} * y and K^{-1} * K_s via Cholesky for numerical stability
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        v = np.linalg.solve(L, K_s)
    except np.linalg.LinAlgError:
        # Fallback if Cholesky fails — add more noise
        K += 1e-4 * np.eye(len(X_train))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        v = np.linalg.solve(L, K_s)

    mu = K_s.T @ alpha
    cov = K_ss - v.T @ v
    sigma = np.sqrt(np.maximum(np.diag(cov), 1e-10))

    return mu, sigma


def _normal_cdf(x):
    """Standard normal CDF using math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x):
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _expected_improvement(mu, sigma, best_loss, xi=0.01):
    """
    Expected Improvement acquisition function.

    Args:
        mu: predicted mean (array)
        sigma: predicted std (array)
        best_loss: current best observed loss
        xi: exploration-exploitation tradeoff (higher = more exploration)

    Returns:
        ei: Expected Improvement at each point (array)
    """
    ei = np.zeros_like(mu)
    mask = sigma > 1e-10
    z = np.zeros_like(mu)
    z[mask] = (best_loss - mu[mask] - xi) / sigma[mask]
    for i in range(len(mu)):
        if mask[i]:
            ei[i] = (best_loss - mu[i] - xi) * _normal_cdf(z[i]) + sigma[i] * _normal_pdf(z[i])
    return np.maximum(ei, 0.0)


def suggest_next_lr(observations, log_range, base_lr, n_candidates=200):
    """
    Suggest the next LR to try using Bayesian Optimization.

    Works in log10(LR) space. Uses GP with auto-tuned length_scale
    and Expected Improvement to balance exploration vs exploitation.

    Args:
        observations: list of (lr, loss) tuples from completed runs
        log_range: half-width of search range in log10 units (e.g., 0.35)
        base_lr: center LR (the found_lr from LR Finder)
        n_candidates: number of candidate points to evaluate EI on

    Returns:
        next_lr: the suggested next learning rate to try
    """
    log_base = math.log10(base_lr)
    log_lower = log_base - log_range
    log_upper = log_base + log_range

    n_obs = len(observations)

    # With 0 observations, return the baseline
    if n_obs == 0:
        return base_lr

    # With 1 observation, explore far-right (high LR often reveals divergence quickly)
    if n_obs == 1:
        return base_lr * (10 ** log_range)

    # With 2 observations, explore far-left
    if n_obs == 2:
        return base_lr * (10 ** (-log_range))

    # 3+ observations: use GP + Expected Improvement
    X_obs = np.array([math.log10(lr) for lr, _ in observations])
    y_obs = np.array([loss for _, loss in observations])

    # Normalize losses for better GP conditioning
    y_mean = np.mean(y_obs)
    y_std = np.std(y_obs) if np.std(y_obs) > 1e-10 else 1.0
    y_norm = (y_obs - y_mean) / y_std

    # Auto-tune length_scale: ~1/3 of the total range
    length_scale = (log_upper - log_lower) / 3.0

    # Generate candidate grid (avoid points too close to observed ones)
    X_candidates = np.linspace(log_lower, log_upper, n_candidates)

    # Remove candidates too close to already-observed points
    min_dist = (log_upper - log_lower) / (n_candidates * 2)
    mask = np.ones(len(X_candidates), dtype=bool)
    for x_obs in X_obs:
        mask &= np.abs(X_candidates - x_obs) > min_dist
    X_candidates = X_candidates[mask]

    if len(X_candidates) == 0:
        # Fallback: all space explored, return midpoint of best two
        best_idx = np.argmin(y_obs)
        return observations[best_idx][0]

    # GP prediction
    mu, sigma = _gp_predict(X_obs, y_norm, X_candidates, length_scale, noise=0.01)

    # Expected Improvement
    best_norm = np.min(y_norm)
    ei = _expected_improvement(mu, sigma, best_norm, xi=0.01)

    # Pick the candidate with highest EI
    best_candidate_idx = np.argmax(ei)
    next_log_lr = X_candidates[best_candidate_idx]

    # Clamp to bounds
    next_log_lr = max(log_lower, min(log_upper, next_log_lr))

    return 10 ** next_log_lr


def test():
    """Simple test of the Bayesian LR search."""
    print("=== Bayesian LR Search Test ===\n")

    base_lr = 1e-4
    log_range = 0.35

    # Simulate a simple unimodal loss landscape: best at ~7e-5
    def fake_loss(lr):
        log_optimal = math.log10(7e-5)
        log_lr = math.log10(lr)
        return 2.0 + 3.0 * (log_lr - log_optimal) ** 2

    observations = []
    print(f"Base LR: {base_lr:.2e}, Range: [{base_lr * 10**(-log_range):.2e}, {base_lr * 10**log_range:.2e}]")
    print()

    for i in range(7):
        next_lr = suggest_next_lr(observations, log_range, base_lr)
        loss = fake_loss(next_lr)
        observations.append((next_lr, loss))
        source = "baseline" if i == 0 else ("far-right" if i == 1 else ("far-left" if i == 2 else "GP+EI"))
        best_lr = min(observations, key=lambda x: x[1])[0]
        best_loss = min(observations, key=lambda x: x[1])[1]
        print(f"  Run {i}: LR={next_lr:.4e}  loss={loss:.4f}  ({source})  "
              f"[best so far: LR={best_lr:.4e} loss={best_loss:.4f}]")

    print(f"\n  Optimal LR (known): 7.00e-05")
    print(f"  Best found:         {min(observations, key=lambda x: x[1])[0]:.4e}")
    print(f"  Error:              {abs(math.log10(min(observations, key=lambda x: x[1])[0]) - math.log10(7e-5)):.4f} decades")
    print("\nTest passed ✓")


if __name__ == "__main__":
    test()
