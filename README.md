# order-book-hawkes-as

Python implementation of three market microstructure components built during an internship project.

## Structure

```
src/
  order_book.py      # LimitOrderBook + LOBGenerator (6-dim Hawkes)
  hawkes.py          # HawkesProcess + MLE calibration
  market_making.py   # AvellanedaStoikov strategy + ASSimulator
notebooks/
  01_order_book.ipynb
  02_hawkes_calibration.ipynb
  03_market_making_as.ipynb
```

## Install

```bash
pip install -r requirements.txt
```

## Quick usage

```python
# LOB
from src.order_book import LimitOrderBook, Side, LOBGenerator

lob = LimitOrderBook(tick_size=0.01)
lob.submit_limit(Side.BID, 99.50, qty=10.0, timestamp=0.0)
lob.submit_limit(Side.ASK, 100.50, qty=10.0, timestamp=0.0)
print(lob.mid_price(), lob.spread())

gen    = LOBGenerator(mid0=100.0, seed=42)
events = gen.generate(T=300.0)

# Hawkes
from src.hawkes import HawkesProcess, calibrate

times  = HawkesProcess(mu=1.0, alpha=0.5, beta=2.5).simulate(T=500.0, seed=0)
result = calibrate(times, T=500.0)
print(result)

# Market making
from src.market_making import AvellanedaStoikov, ASSimulator

strat  = AvellanedaStoikov(gamma=0.01, sigma=0.5, kappa=1.5, T=300.0)
gen    = LOBGenerator(mid0=100.0)
result = ASSimulator(strat, gen, T=300.0).run()
print(result.summary())
```

## Model notes

**Hawkes process** — λ*(t) = μ + Σ α·exp(−β(t−tᵢ))  
Simulated with Ogata's thinning. MLE uses the Ozaki (1979) recursive formula for O(n) log-likelihood.  
Stationarity condition: α/β < 1.

**Avellaneda-Stoikov (2008)** — reservation price r = S − qγσ²(T−t),  
optimal half-spread δ* = γσ²(T−t)/2 + (1/γ)·ln(1+γ/κ).  
The LOB generator drives order arrivals with a 6-dim Hawkes process (limit/market/cancel on each side).

## References

- Hawkes (1971), Biometrika
- Ozaki (1979), Ann. Inst. Stat. Math.
- Avellaneda & Stoikov (2008), Quantitative Finance
- Bacry, Mastromatteo & Muzy (2015), Market Microstructure and Liquidity
