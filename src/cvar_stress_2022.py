"""CVaR portfolio analysis for a stressed market period.

This script uses the same methodology and output structure as `cvar_real.py`,
but restricts the historical sample to a period of higher market instability:
from 2021-11-01 to 2022-12-31.

Outputs:
- portfolio weights for CVaR, Equal Weight, and Aggressive portfolios;
- train/test VaR and CVaR interpretation for the optimized CVaR portfolio;
- train/test summary tables with the same metrics as the main script;
- one chart comparing portfolio weights;
- one chart comparing train/test equity curves.
"""

from __future__ import annotations

import pandas as pd

try:
    from cvar_real import (
        ALPHA,
        LONG_ONLY,
        MAX_WEIGHT,
        TRAIN_RATIO,
        USE_MONTHLY_RETURNS,
        build_aggressive_portfolio,
        build_equal_weight_portfolio,
        build_metrics_table,
        compute_returns,
        download_prices,
        historical_var_cvar,
        optimize_cvar_portfolio,
        plot_results,
        portfolio_returns,
        print_metrics,
        print_summary_table,
        to_equity,
    )
except ModuleNotFoundError:
    # Local preview fallback. In the GitHub repo this script will live in `src/`
    # next to `cvar_real.py`, so the import above will be the normal path.
    from cvar_real_refined import (
        ALPHA,
        LONG_ONLY,
        MAX_WEIGHT,
        TRAIN_RATIO,
        USE_MONTHLY_RETURNS,
        build_aggressive_portfolio,
        build_equal_weight_portfolio,
        build_metrics_table,
        compute_returns,
        download_prices,
        historical_var_cvar,
        optimize_cvar_portfolio,
        plot_results,
        portfolio_returns,
        print_metrics,
        print_summary_table,
        to_equity,
    )


# =========================
# Stress-period configuration
# =========================

TICKERS = ["SPY", "QQQ", "TLT", "GLD", "EEM", "VNQ"]
START_DATE = "2021-11-01"
END_DATE = "2022-12-31"
PERIOD_LABEL = "2021-11-01 to 2022-12-31"


def build_equity_curves(
    cvar_train_returns: pd.Series,
    cvar_test_returns: pd.Series,
    eq_train_returns: pd.Series,
    eq_test_returns: pd.Series,
    aggressive_train_returns: pd.Series,
    aggressive_test_returns: pd.Series,
) -> dict[str, pd.Series]:
    """Create train/test equity curves for the three portfolios."""

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

    return {
        "CVaR Train": cvar_train_equity,
        "CVaR Test": cvar_test_equity,
        "Equal Weight Train": eq_train_equity,
        "Equal Weight Test": eq_test_equity,
        "Aggressive Train": aggressive_train_equity,
        "Aggressive Test": aggressive_test_equity,
    }


def main() -> None:
    print(f"Descargando datos para periodo de estrés: {PERIOD_LABEL}...")

    prices = download_prices(TICKERS, START_DATE, END_DATE)
    returns, freq = compute_returns(prices, USE_MONTHLY_RETURNS)
    train_ret, test_ret = split_returns_for_stress_period(returns)

    print("\nActivos usados:")
    print(", ".join(train_ret.columns.tolist()))
    print(f"\nPeriodo analizado:    {PERIOD_LABEL}")
    print(f"Observaciones train: {len(train_ret)}")
    print(f"Observaciones test:  {len(test_ret)}")
    print(f"Frecuencia:          {freq}")

    # Build the same three portfolios used in the main comparison script.
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

    equity_curves = build_equity_curves(
        cvar_train_returns=cvar_train_returns,
        cvar_test_returns=cvar_test_returns,
        eq_train_returns=eq_train_returns,
        eq_test_returns=eq_test_returns,
        aggressive_train_returns=aggressive_train_returns,
        aggressive_test_returns=aggressive_test_returns,
    )

    plot_results(
        weights_table=weights_table,
        equity_curves=equity_curves,
        split_date=test_ret.index[0],
    )


def split_returns_for_stress_period(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the same train/test split logic as the main script."""

    split_idx = int(len(returns) * TRAIN_RATIO)
    train = returns.iloc[:split_idx].copy()
    test = returns.iloc[split_idx:].copy()

    if len(train) < 50 or len(test) < 20:
        raise RuntimeError("Muy pocos datos para train/test. Ajusta fechas o frecuencia.")

    return train, test


if __name__ == "__main__":
    main()
