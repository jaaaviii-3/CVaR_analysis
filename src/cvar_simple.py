"""Simple CVaR portfolio optimization example.

This script is intentionally small. It uses simulated returns to show the
mathematical structure of a CVaR minimization problem before moving to real
market data.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


N_SCENARIOS = 2_500
N_ASSETS = 100
ALPHA = 0.95
RANDOM_SEED = 42


def simulate_returns(
    n_scenarios: int = N_SCENARIOS,
    n_assets: int = N_ASSETS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate synthetic daily returns for a group of assets."""

    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0003, scale=0.01, size=(n_scenarios, n_assets))


def optimize_cvar(
    returns: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, float, float]:
    """Minimize historical CVaR for a long-only fully invested portfolio."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    n_scenarios, n_assets = returns.shape

    weights = cp.Variable(n_assets)
    gamma = cp.Variable()

    portfolio_losses = -(returns @ weights)
    tail_losses = cp.pos(portfolio_losses - gamma)

    cvar = gamma + (1 / ((1 - alpha) * n_scenarios)) * cp.sum(tail_losses)

    constraints = [
        weights >= 0,
        cp.sum(weights) == 1,
    ]

    problem = cp.Problem(cp.Minimize(cvar), constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Optimization failed with status: {problem.status}")

    if weights.value is None or gamma.value is None or problem.value is None:
        raise RuntimeError("Optimization finished without a valid solution.")

    optimized_weights = np.asarray(weights.value).ravel()
    optimized_weights = np.maximum(optimized_weights, 0)
    optimized_weights = optimized_weights / optimized_weights.sum()

    return optimized_weights, float(gamma.value), float(problem.value)


def historical_var_cvar(
    portfolio_returns: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    """Calculate historical VaR and CVaR from realized portfolio returns."""

    losses = -portfolio_returns
    var_alpha = np.quantile(losses, alpha)
    cvar_alpha = losses[losses >= var_alpha].mean()

    return float(var_alpha), float(cvar_alpha)


def main() -> None:
    returns = simulate_returns()
    weights, gamma, optimized_cvar = optimize_cvar(returns, ALPHA)

    portfolio_returns = returns @ weights
    realized_var, realized_cvar = historical_var_cvar(portfolio_returns, ALPHA)

    top_positions = np.argsort(weights)[::-1][:10]

    print("\n=== Simple CVaR Optimization Example ===")
    print(f"Scenarios: {returns.shape[0]}")
    print(f"Assets:    {returns.shape[1]}")
    print(f"Alpha:     {ALPHA:.2f}")

    print("\n=== Optimization Results ===")
    print(f"Gamma / VaR threshold: {gamma:.4%}")
    print(f"Optimized CVaR:        {optimized_cvar:.4%}")
    print(f"Historical VaR:        {realized_var:.4%}")
    print(f"Historical CVaR:       {realized_cvar:.4%}")

    print("\n=== Top 10 Portfolio Weights ===")
    for asset_idx in top_positions:
        print(f"Asset {asset_idx:03d}: {weights[asset_idx]:.2%}")


if __name__ == "__main__":
    main()
