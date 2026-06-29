"""CVaR portfolio backtest with three portfolio comparisons.

The script keeps the original methodology:
- download historical prices with yfinance;
- calculate daily or monthly returns;
- split the sample into train and test using a fixed ratio;
- optimize a long-only CVaR portfolio on the train sample;
- compare it with an equal-weight portfolio and a manual aggressive portfolio;
- print summary metrics and plot portfolio weights plus equity curves.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


warnings.filterwarnings("ignore")


# =========================
# Configuration
# =========================

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "EEM", "VNQ"]
START_DATE = "2018-01-01"
END_DATE = None                 # None means use the latest available data.
ALPHA = 0.95                    # Confidence level for VaR / CVaR.
TRAIN_RATIO = 0.7               # 70% train, 30% test.
LONG_ONLY = True                # No short positions.
MAX_WEIGHT = 0.45               # Maximum weight per asset in the CVaR portfolio.
USE_MONTHLY_RETURNS = False     # False = daily returns, True = monthly returns.


# =========================
# Metrics and utilities
# =========================

MetricFormatter = dict[str, Callable[[float], str]]


def annualization_factor(freq: str) -> int:
    """Return the annualization factor for the selected return frequency."""

    if freq == "daily":
        return 252
    if freq == "monthly":
        return 12
    raise ValueError("Frecuencia no soportada")


def max_drawdown(equity: pd.Series) -> float:
    """Calculate the maximum drawdown of an equity curve."""

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def sortino_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Calculate the annualized Sortino ratio."""

    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    downside = returns[returns < 0]
    downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    mean_ret = returns.mean()

    if downside_dev == 0:
        if mean_ret > 0:
            return np.inf
        if mean_ret < 0:
            return -np.inf
        return 0.0

    return float((mean_ret / downside_dev) * np.sqrt(periods_per_year))


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    """Calculate compound annual growth rate from an equity curve."""

    if len(equity) < 2:
        return np.nan

    total_return = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / periods_per_year

    if years <= 0:
        return np.nan

    return float(total_return ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Calculate annualized volatility."""

    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    return float(returns.std() * np.sqrt(periods_per_year))


def to_equity(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Convert a return series into an equity curve."""

    return initial_value * (1 + returns).cumprod()


def historical_var_cvar(returns: pd.Series, alpha: float) -> tuple[float, float]:
    """Calculate historical VaR and CVaR over portfolio losses."""

    losses = -returns.dropna()
    var_alpha = np.quantile(losses, alpha)
    tail_losses = losses[losses >= var_alpha]
    cvar_alpha = tail_losses.mean() if len(tail_losses) else np.nan

    return float(var_alpha), float(cvar_alpha)


def portfolio_metrics(
    name: str,
    returns: pd.Series,
    freq: str,
    alpha: float,
) -> dict[str, float | str]:
    """Build one row of the portfolio metrics summary table."""

    periods = annualization_factor(freq)
    equity = to_equity(returns)
    var_alpha, cvar_alpha = historical_var_cvar(returns, alpha)

    return {
        "Portfolio": name,
        "Total Return": equity.iloc[-1] - 1.0,
        "CAGR": cagr(equity, periods),
        "Volatility": annualized_volatility(returns, periods),
        "Sortino": sortino_ratio(returns, periods),
        "Max Drawdown": max_drawdown(equity),
        "VaR": var_alpha,
        "CVaR": cvar_alpha,
    }


def print_metrics(name: str, returns: pd.Series, freq: str) -> None:
    """Print the same individual metric block used in the original script."""

    periods = annualization_factor(freq)
    equity = to_equity(returns)
    total_return = equity.iloc[-1] - 1.0

    print(f"\n=== {name} ===")
    print(f"Retorno total:   {total_return:.2%}")
    print(f"CAGR:            {cagr(equity, periods):.2%}")
    print(f"Volatilidad:     {annualized_volatility(returns, periods):.2%}")
    print(f"Sortino:         {sortino_ratio(returns, periods):.3f}")
    print(f"Max drawdown:    {max_drawdown(equity):.2%}")


def metrics_formatters() -> MetricFormatter:
    """Return consistent formatting for the train/test summary tables."""

    return {
        "Total Return": "{:.2%}".format,
        "CAGR": "{:.2%}".format,
        "Volatility": "{:.2%}".format,
        "Sortino": "{:.3f}".format,
        "Max Drawdown": "{:.2%}".format,
        "VaR": "{:.2%}".format,
        "CVaR": "{:.2%}".format,
    }


def print_summary_table(title: str, metrics: pd.DataFrame) -> None:
    """Print a formatted summary table in the terminal."""

    print(f"\n=== {title} ===")
    print(
        metrics.to_string(
            index=False,
            formatters=metrics_formatters(),
        )
    )


# =========================
# Data
# =========================

def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices and apply basic cleaning."""

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise RuntimeError("No se pudieron descargar datos.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data.copy()

    prices = prices.dropna(how="all")
    prices = prices.ffill().dropna()

    if prices.empty:
        raise RuntimeError("Precios vacíos después de limpiar datos.")

    return prices


def compute_returns(
    prices: pd.DataFrame,
    use_monthly: bool,
) -> tuple[pd.DataFrame, str]:
    """Calculate daily or monthly percentage returns."""

    if use_monthly:
        monthly_prices = prices.resample("M").last()
        returns = monthly_prices.pct_change().dropna()
        return returns, "monthly"

    returns = prices.pct_change().dropna()
    return returns, "daily"


def split_train_test(
    returns: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split returns into train and test using the original ratio logic."""

    split_idx = int(len(returns) * train_ratio)

    train = returns.iloc[:split_idx].copy()
    test = returns.iloc[split_idx:].copy()

    if len(train) < 50 or len(test) < 20:
        raise RuntimeError("Muy pocos datos para train/test. Ajusta fechas o frecuencia.")

    return train, test


# =========================
# CVaR optimization
# =========================

def optimize_cvar_portfolio(
    train_returns: pd.DataFrame,
    alpha: float = 0.95,
    long_only: bool = True,
    max_weight: float | None = None,
) -> tuple[pd.Series, float, float]:
    """Minimize historical CVaR using portfolio losses."""

    returns_matrix = train_returns.values
    n_scenarios, n_assets = returns_matrix.shape

    weights = cp.Variable(n_assets)
    gamma = cp.Variable()

    losses = -(returns_matrix @ weights)
    cvar = gamma + (1 / ((1 - alpha) * n_scenarios)) * cp.sum(
        cp.pos(losses - gamma)
    )

    constraints = [cp.sum(weights) == 1]

    if long_only:
        constraints.append(weights >= 0)

    if max_weight is not None:
        constraints.append(weights <= max_weight)

    problem = cp.Problem(cp.Minimize(cvar), constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"No se pudo resolver el problema: {problem.status}")

    if weights.value is None or gamma.value is None or problem.value is None:
        raise RuntimeError("No se pudo resolver el problema de optimización.")

    optimized_weights = pd.Series(
        np.array(weights.value).flatten(),
        index=train_returns.columns,
    )
    optimized_weights = optimized_weights / optimized_weights.sum()

    return optimized_weights, float(gamma.value), float(problem.value)


# =========================
# Portfolio construction
# =========================

def portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Calculate portfolio returns after aligning weights to return columns."""

    aligned_weights = weights.reindex(returns.columns).fillna(0.0)
    return returns @ aligned_weights


def build_equal_weight_portfolio(columns: pd.Index) -> pd.Series:
    """Build an equal-weight portfolio with the same assets."""

    n_assets = len(columns)
    return pd.Series(np.repeat(1 / n_assets, n_assets), index=columns)


def build_aggressive_portfolio(columns: pd.Index) -> pd.Series:
    """Build the manual aggressive benchmark portfolio."""

    aggressive_weights = {
        "SPY": 0.30,
        "QQQ": 0.45,
        "TLT": 0.00,
        "GLD": 0.00,
        "EEM": 0.15,
        "VNQ": 0.10,
    }

    weights = pd.Series(aggressive_weights)
    weights = weights.reindex(columns).fillna(0.0)

    if not np.isclose(weights.sum(), 1.0):
        weights = weights / weights.sum()

    return weights


def build_metrics_table(
    portfolio_return_map: dict[str, pd.Series],
    freq: str,
    alpha: float,
) -> pd.DataFrame:
    """Create a metrics table for all portfolios in a given period."""

    return pd.DataFrame(
        [
            portfolio_metrics(name, returns, freq, alpha)
            for name, returns in portfolio_return_map.items()
        ]
    )


# =========================
# Plotting
# =========================

def plot_results(
    weights_table: pd.DataFrame,
    equity_curves: dict[str, pd.Series],
    split_date: pd.Timestamp,
) -> None:
    """Plot portfolio weights and train/test equity curves."""

    weights_table = weights_table.copy()
    weights_table.index.name = "Ticker"

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 1. Portfolio weights for the three strategies.
    weights_table.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Comparación de pesos por cartera")
    axes[0].set_ylabel("Peso")
    axes[0].set_xlabel("Ticker")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    # 2. Equity curves, keeping train and test visually separated.
    axes[1].plot(
        equity_curves["CVaR Train"].index,
        equity_curves["CVaR Train"].values,
        label="CVaR Train",
    )
    axes[1].plot(
        equity_curves["CVaR Test"].index,
        equity_curves["CVaR Test"].values,
        label="CVaR Test",
    )
    axes[1].plot(
        equity_curves["Equal Weight Train"].index,
        equity_curves["Equal Weight Train"].values,
        label="Equal Weight Train",
        linestyle="--",
    )
    axes[1].plot(
        equity_curves["Equal Weight Test"].index,
        equity_curves["Equal Weight Test"].values,
        label="Equal Weight Test",
        linestyle="--",
    )
    axes[1].plot(
        equity_curves["Aggressive Train"].index,
        equity_curves["Aggressive Train"].values,
        label="Aggressive Train",
        linestyle="-.",
    )
    axes[1].plot(
        equity_curves["Aggressive Test"].index,
        equity_curves["Aggressive Test"].values,
        label="Aggressive Test",
        linestyle="-.",
    )
    axes[1].axvline(
        split_date,
        color="black",
        linestyle=":",
        label="Inicio Test",
    )

    axes[1].set_title("Comparación de carteras: CVaR vs equiponderada vs agresiva")
    axes[1].set_ylabel("Equity")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# =========================
# Main workflow
# =========================

def main() -> None:
    print("Descargando datos...")

    prices = download_prices(TICKERS, START_DATE, END_DATE)
    returns, freq = compute_returns(prices, USE_MONTHLY_RETURNS)
    train_ret, test_ret = split_train_test(returns, TRAIN_RATIO)

    print("\nActivos usados:")
    print(", ".join(train_ret.columns.tolist()))
    print(f"\nObservaciones train: {len(train_ret)}")
    print(f"Observaciones test:  {len(test_ret)}")
    print(f"Frecuencia:          {freq}")

    # Build the three portfolios: optimized CVaR, equal weight, and aggressive.
    weights_cvar, gamma_train, cvar_opt_train = optimize_cvar_portfolio(
        train_returns=train_ret,
        alpha=ALPHA,
        long_only=LONG_ONLY,
        max_weight=MAX_WEIGHT,
    )
    weights_eq = build_equal_weight_portfolio(train_ret.columns)
    weights_aggressive = build_aggressive_portfolio(train_ret.columns)

    # Calculate train/test returns for each portfolio.
    cvar_train_returns = portfolio_returns(train_ret, weights_cvar)
    cvar_test_returns = portfolio_returns(test_ret, weights_cvar)

    eq_train_returns = portfolio_returns(train_ret, weights_eq)
    eq_test_returns = portfolio_returns(test_ret, weights_eq)

    aggressive_train_returns = portfolio_returns(train_ret, weights_aggressive)
    aggressive_test_returns = portfolio_returns(test_ret, weights_aggressive)

    # Historical VaR/CVaR interpretation for the optimized CVaR portfolio.
    var_train_real, cvar_train_real = historical_var_cvar(cvar_train_returns, ALPHA)
    var_test_real, cvar_test_real = historical_var_cvar(cvar_test_returns, ALPHA)

    # Equity curves. Test curves start from the final train equity value.
    cvar_train_equity = to_equity(cvar_train_returns)
    cvar_test_equity = to_equity(
        cvar_test_returns,
        initial_value=float(cvar_train_equity.iloc[-1]),
    )

    eq_train_equity = to_equity(eq_train_returns)
    eq_test_equity = to_equity(
        eq_test_returns,
        initial_value=float(eq_train_equity.iloc[-1]),
    )

    aggressive_train_equity = to_equity(aggressive_train_returns)
    aggressive_test_equity = to_equity(
        aggressive_test_returns,
        initial_value=float(aggressive_train_equity.iloc[-1]),
    )

    weights_table = pd.DataFrame(
        {
            "CVaR": weights_cvar,
            "Equal Weight": weights_eq,
            "Aggressive": weights_aggressive,
        }
    )

    print("\n=== PESOS DE LAS CARTERAS ===")
    print((weights_table * 100).round(2).astype(str) + "%")

    print("\n=== INTERPRETACIÓN TRAIN CVaR ===")
    print(f"Gamma (umbral tipo VaR de la optimización): {gamma_train:.2%}")
    print(f"CVaR óptimo del problema:                   {cvar_opt_train:.2%}")
    print(f"VaR histórico de la cartera CVaR:          {var_train_real:.2%}")
    print(f"CVaR histórico de la cartera CVaR:         {cvar_train_real:.2%}")

    print("\n=== INTERPRETACIÓN TEST CVaR ===")
    print(f"VaR histórico de la cartera CVaR:          {var_test_real:.2%}")
    print(f"CVaR histórico de la cartera CVaR:         {cvar_test_real:.2%}")

    print_metrics("Cartera CVaR - TRAIN", cvar_train_returns, freq)
    print_metrics("Cartera CVaR - TEST", cvar_test_returns, freq)

    print_metrics("Cartera equiponderada - TRAIN", eq_train_returns, freq)
    print_metrics("Cartera equiponderada - TEST", eq_test_returns, freq)

    print_metrics("Cartera agresiva - TRAIN", aggressive_train_returns, freq)
    print_metrics("Cartera agresiva - TEST", aggressive_test_returns, freq)

    train_portfolios = {
        "CVaR": cvar_train_returns,
        "Equal Weight": eq_train_returns,
        "Aggressive": aggressive_train_returns,
    }
    test_portfolios = {
        "CVaR": cvar_test_returns,
        "Equal Weight": eq_test_returns,
        "Aggressive": aggressive_test_returns,
    }

    metrics_train = build_metrics_table(train_portfolios, freq, ALPHA)
    metrics_test = build_metrics_table(test_portfolios, freq, ALPHA)

    print_summary_table("TABLA RESUMEN TRAIN", metrics_train)
    print_summary_table("TABLA RESUMEN TEST", metrics_test)

    equity_curves = {
        "CVaR Train": cvar_train_equity,
        "CVaR Test": cvar_test_equity,
        "Equal Weight Train": eq_train_equity,
        "Equal Weight Test": eq_test_equity,
        "Aggressive Train": aggressive_train_equity,
        "Aggressive Test": aggressive_test_equity,
    }

    plot_results(
        weights_table=weights_table,
        equity_curves=equity_curves,
        split_date=test_ret.index[0],
    )


if __name__ == "__main__":
    main()
