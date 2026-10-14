# Order Book Analysis

This project implements market microstructure models I put together while exploring quantitative finance. The goal was to understand how limit order books actually work, how order flow can be modeled, and how a market maker should behave optimally.

## About this project

I'm a student at CentraleSupélec. I built this to get a concrete understanding of the models behind market making and high-frequency trading. The three components — order book, Hawkes calibration, and Avellaneda-Stoikov strategy — fit together into a small simulation framework.

## What's implemented

**Limit Order Book** — price-time priority matching engine with limit, market and cancel orders. Order flow is driven by a 6-dimensional Hawkes process (limit/market/cancel on each side). Limit orders are placed around a latent mid; an optional `impact` parameter makes market orders move that mid (Kyle-style permanent impact), so that order flow carries information.

**Hawkes Process** — univariate simulation and MLE calibration, parametrised by (μ, n = α/β, β) so that stationarity (n < 1) is imposed directly. Log-likelihood computed in O(n) using the Ozaki (1979) recursive formula. Goodness-of-fit via Ogata (1988) residuals.

**Avellaneda-Stoikov (2008)** — optimal market making under inventory risk, using the closed-form approximation of the paper for the reservation price and spread. Simulated against the LOB event by event. σ and κ are estimated on the simulated order flow (realised variance of the mid; log-linear fit of the rate at which market orders reach a distance δ). Results are compared over 200 seeds, and adverse selection is measured with markouts of the passive fills.

## Project structure

```
src/
  order_book.py      # LimitOrderBook + LOBGenerator (6-dim Hawkes)
  hawkes.py          # HawkesProcess, MLE calibration, moment estimator
  market_making.py   # AvellanedaStoikov strategy + ASSimulator
notebooks/
  01_order_book.ipynb           # LOB dynamics and order flow statistics
  02_hawkes_calibration.ipynb   # simulation, MLE, GoF, identifiability
  03_market_making_as.ipynb     # AS strategy, P&L decomposition, gamma sweep
```

## Getting started

```bash
git clone https://github.com/Alexandre-Reyob/order-book-analysis.git
cd order-book-analysis
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Model notes

**Hawkes process** — λ*(t) = μ + Σ α·exp(−β(t−tᵢ))
Simulated with Ogata's thinning. Stationarity requires n = α/β < 1. The estimates of α and β are strongly correlated, so n is estimated much more precisely than α and β individually, which converge slowly on short samples.

**Avellaneda-Stoikov (2008)** — reservation price r = S − qγσ²(T−t),
optimal half-spread δ* = γσ²(T−t)/2 + (1/γ)·ln(1+γ/κ).
The reservation price moves against the inventory, so both quotes shift to let the order flow bring the inventory back to zero.

Main findings of notebook 03 (simulated market, 200 seeds per setting):
- risk aversion trades fills against inventory control; for large γ the inventory skew is so large that the unwinding quote crosses the book, the strategy pays the spread and P&L turns negative (the original model has no inventory bounds);
- with `impact = 0` markouts are flat (no adverse selection); with `impact > 0` they drop right after the fill and the P&L of the calibrated strategy falls quickly, since Avellaneda-Stoikov assumes a mid independent of order flow.

## About me

I'm Alexandre Boyer, a student at CentraleSupélec in the Mathematical and Financial Modelling track. I built this project to go deeper into the microstructure models I come across during my studies. It's a learning project, not production code.

## References

- Hawkes (1971), *Spectra of some self-exciting and mutually exciting point processes*, Biometrika
- Ozaki (1979), *Maximum likelihood estimation of Hawkes' self-exciting point processes*, Ann. Inst. Stat. Math.
- Ogata (1988), *Statistical models for earthquake occurrences*, J. Amer. Stat. Assoc.
- Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*, Quantitative Finance
- Bacry, Mastromatteo & Muzy (2015), *Hawkes processes in finance*, Market Microstructure and Liquidity
