# CVaR Analysis

This project is based on building a portfolio of financial assets by minimizing the extreme risk that the portfolio can suffer in its worst scenarios. The central risk measure used in the project is CVaR, also known as Conditional Value at Risk or Expected Shortfall. The financial idea behind this approach is that it is not enough to look only at volatility or variance, because those measures treat positive and negative movements in a similar way and do not focus specifically on severe losses. CVaR, instead, looks directly at what happens in the bad tail of the return distribution.

## Financial intuition

To understand the project, it is useful to distinguish first between VaR and CVaR. VaR at a 95 percent confidence level defines a loss threshold such that, in 95 percent of the observed scenarios, the loss does not exceed that value. In simple terms, VaR marks where the bad part of the distribution begins.

CVaR goes one step further. It measures the average loss inside the worst 5 percent of scenarios. Put differently, VaR tells us where the tail starts, while CVaR tells us how damaging that tail is on average. This distinction is important because two portfolios can have a similar VaR but very different losses once the market enters the most adverse scenarios.

The objective function of the optimization problem is built to minimize that average tail loss. For that purpose, the formulation includes an auxiliary variable, usually called gamma, which acts as a VaR-like threshold. The parameter alpha defines which fraction of scenarios is considered extreme. For example, when alpha is equal to 0.95, the model focuses on the worst 5 percent of portfolio outcomes. The optimization then measures how much the losses exceed the gamma threshold and minimizes the average excess loss beyond that point.

The main advantage of this formulation is that CVaR portfolio optimization can be written as a convex optimization problem. This means that, under the chosen constraints, the solution found by the optimizer is a global optimum rather than a local minimum. From a financial perspective, this is useful because a portfolio may look reasonable on average while still hiding very damaging tail scenarios.

## Project objective

The goal of this project is to find a realistic allocation of capital across several assets, not by simply looking for the highest return, but by trying to reduce the damage suffered in the worst market scenarios. The project starts from a matrix of asset returns, where each column represents one asset and each row represents one historical observation or scenario.

From those returns, the model defines a vector of portfolio weights and the gamma threshold. The optimizer chooses the weights that minimize CVaR, which represents the expected loss in the worst part of the return distribution. Basic constraints are added to make the portfolio more realistic: the weights must sum to one, the allocation can be long-only, and a maximum weight per asset can be imposed to avoid excessive concentration.

The logic of the project is therefore simple but powerful: first, calculate portfolio returns across many scenarios; second, identify the relevant loss threshold; third, minimize the average loss beyond that threshold. The project is not trying to predict which asset will rise the most. It is trying to allocate capital in a way that makes the portfolio more resilient when severe market conditions appear.

## Initial script

The first version of the work is intended as a conceptual implementation of CVaR optimization in Python. It uses tools such as NumPy and CVXPY to make the mathematical structure of the problem explicit. In this initial stage, the return matrix can be simulated or simplified so that the focus remains on understanding the role of each element: the return matrix as a set of scenarios, the portfolio weights as investment decisions, gamma as a VaR-like threshold, and the final objective value as the optimized CVaR.

This simple version is useful because it explains the mechanics of the model before moving to real market data. It shows how the optimizer assigns weights, how the constraints affect the solution, and how the CVaR value can be interpreted as a measure of average loss in the worst scenarios.

## Realistic script

The next version of the project applies the same CVaR logic to real financial data. Instead of using simulated returns, the script downloads historical prices for a set of assets, such as ETFs representing different market exposures. Prices are transformed into returns, and the dataset is split into a training period and a test period.

The training period is used to construct the CVaR-optimized portfolio. The test period is then used to evaluate how that portfolio behaves out of sample, which is important because a portfolio that works only on the data used to build it may not be robust. The CVaR portfolio can then be compared with simple benchmark portfolios, such as an equal-weight portfolio or a more aggressive manually defined allocation.

After the optimal weights are obtained, the project calculates portfolio returns and equity curves for both train and test periods. It also computes financial metrics such as total return, CAGR, annualized volatility, Sortino ratio, maximum drawdown, VaR, and CVaR. This allows the analysis to compare not only which portfolio earns more, but also which portfolio behaves better from a risk perspective.

## Results and interpretation

A CVaR-optimized portfolio may underperform an equal-weight or aggressive portfolio during favorable market periods. This does not necessarily mean that the CVaR approach is wrong. It usually means that the optimized portfolio is more defensive and may sacrifice part of the upside in exchange for better protection against severe losses.

For example, if the CVaR portfolio assigns more weight to defensive assets such as bonds or gold, it may lag behind equity-heavy portfolios during strong bull markets. However, it may also experience a lower maximum drawdown or lower average tail loss during difficult periods. The key question is not only which portfolio grows faster, but whether the reduction in extreme risk compensates for the potential loss of return.

This project therefore focuses on understanding the trade-off between return and tail-risk protection. A portfolio with lower CVaR is not automatically better in every context, but it can be more suitable for investors or strategies that care especially about severe downside scenarios.

## Planned improvements

The project is intended to evolve toward more advanced and realistic versions. Some natural next steps are:

- Use more specific tickers and asset universes.
- Test different historical periods, including crisis and non-crisis regimes.
- Compare daily, weekly, and monthly returns.
- Add rolling or walk-forward optimization.
- Include transaction costs and rebalancing frequency.
- Add a minimum expected return constraint.
- Compare the CVaR portfolio with several benchmarks.
- Store results in tables and charts for easier analysis.
- Split the code into reusable modules instead of keeping everything in one script.

## Suggested project structure

The repository is expected to be organized in a modular way as the project grows:

```text
CVaR_analysis/
|-- README.md
|-- .gitignore
|-- requirements.txt
|-- config.yaml
|-- src/
|   |-- data.py
|   |-- metrics.py
|   |-- optimization.py
|   |-- portfolios.py
|   |-- plotting.py
|   |-- run_backtest.py
|-- notebooks/
|   |-- cvar_portfolio_exploration.ipynb
|-- outputs/
|-- tests/
```

A modular structure makes the project easier to extend. Data downloading, metric calculation, CVaR optimization, portfolio construction, plotting, and the main backtest logic can each live in a separate file. This is preferable to keeping every idea in a single large script, especially if the project will later test different tickers, periods, frequencies, and portfolio constraints.

## Main dependencies

The project will mainly rely on:

- `numpy`
- `pandas`
- `cvxpy`
- `matplotlib`
- `yfinance`
- `pyyaml`

## Disclaimer

This repository is for educational and research purposes only. It does not provide investment advice, trading recommendations, or a guarantee of future performance. Financial markets involve risk, and historical results do not ensure similar behavior in the future.
