from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# Avellaneda & Stoikov (2008) optimal market making
# r = S - q*gamma*sigma^2*(T-t)
# delta* = gamma*sigma^2*(T-t)/2 + (1/gamma)*ln(1 + gamma/kappa)
# bid* = r - delta*,  ask* = r + delta*

@dataclass
class Quote:
    timestamp:         float
    mid:               float
    reservation_price: float
    half_spread:       float
    bid:               float
    ask:               float
    inventory:         float
    time_remaining:    float

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def __repr__(self) -> str:
        return (f"Quote(t={self.timestamp:.3f}  mid={self.mid:.4f}  r={self.reservation_price:.4f}  "
                f"bid={self.bid:.4f}  ask={self.ask:.4f}  spread={self.spread:.4f}  q={self.inventory:.1f})")


class AvellanedaStoikov:

    def __init__(self, gamma: float = 0.1, sigma: float = 2.0, kappa: float = 1.5,
                 T: float = 1.0, max_inventory: float = 50.0):
        if gamma <= 0: raise ValueError("gamma must be > 0")
        if sigma <= 0: raise ValueError("sigma must be > 0")
        if kappa <= 0: raise ValueError("kappa must be > 0")
        self.gamma         = gamma
        self.sigma         = sigma
        self.kappa         = kappa
        self.T             = T
        self.max_inventory = max_inventory

    def reservation_price(self, S: float, q: float, t: float) -> float:
        return S - q * self.gamma * self.sigma**2 * max(self.T - t, 0.0)

    def optimal_half_spread(self, t: float) -> float:
        tau = max(self.T - t, 0.0)
        return 0.5 * self.gamma * self.sigma**2 * tau + (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)

    def quote(self, S: float, q: float, t: float) -> Quote:
        r     = self.reservation_price(S, q, t)
        delta = self.optimal_half_spread(t)
        return Quote(timestamp=t, mid=S, reservation_price=r, half_spread=delta,
                     bid=r-delta, ask=r+delta, inventory=q, time_remaining=max(self.T-t, 0.0))

    def skewed_quote(self, S: float, q: float, t: float) -> Quote:
        # tilt quotes toward inventory unwind when close to the limit
        base = self.quote(S, q, t)
        adj  = np.clip(q / self.max_inventory, -1.0, 1.0) * base.half_spread * 0.5
        return Quote(timestamp=t, mid=S, reservation_price=base.reservation_price,
                     half_spread=base.half_spread, bid=base.bid-adj, ask=base.ask-adj,
                     inventory=q, time_remaining=base.time_remaining)

    def arrival_intensity(self, delta: float, A: float = 1.0) -> float:
        return A * np.exp(-self.kappa * delta)

    @property
    def spread_t0(self) -> float:
        return 2.0 * self.optimal_half_spread(0.0)

    @property
    def spread_terminal(self) -> float:
        return 2.0 * (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)

    def __repr__(self) -> str:
        return (f"AvellanedaStoikov(gamma={self.gamma}, sigma={self.sigma}, kappa={self.kappa}, T={self.T})\n"
                f"  spread t=0: {self.spread_t0:.5f}   spread t=T: {self.spread_terminal:.5f}")


@dataclass
class SimStep:
    timestamp:  float
    mid:        float
    inventory:  float
    cash:       float
    pnl:        float
    quote:      Quote
    bid_filled: float
    ask_filled: float


@dataclass
class SimulationResult:
    steps: List[SimStep] = field(default_factory=list)

    final_pnl:     float = 0.0
    max_inventory: float = 0.0
    n_fills:       int   = 0
    sharpe:        float = 0.0

    def finalise(self):
        if not self.steps: return
        pnls              = np.array([s.pnl for s in self.steps])
        self.final_pnl    = float(pnls[-1])
        self.max_inventory= float(max(abs(s.inventory) for s in self.steps))
        self.n_fills      = sum(1 for s in self.steps if s.bid_filled + s.ask_filled > 0)
        if len(pnls) > 1:
            ret = np.diff(pnls)
            std = ret.std()
            # annualized with 252 days -- convention, un peu arbitraire
            self.sharpe = float(ret.mean() / std * np.sqrt(252)) if std > 1e-10 else 0.0

    def pnl_series(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([s.timestamp for s in self.steps]), np.array([s.pnl for s in self.steps])

    def inventory_series(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([s.timestamp for s in self.steps]), np.array([s.inventory for s in self.steps])

    def spread_series(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([s.timestamp for s in self.steps]), np.array([s.quote.spread for s in self.steps])

    def mid_series(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([s.timestamp for s in self.steps]), np.array([s.mid for s in self.steps])

    def bid_ask_series(self):
        t    = np.array([s.timestamp  for s in self.steps])
        bids = np.array([s.quote.bid  for s in self.steps])
        asks = np.array([s.quote.ask  for s in self.steps])
        return t, bids, asks

    def summary(self) -> str:
        return (f"final PnL={self.final_pnl:+.4f}  |inv|_max={self.max_inventory:.1f}  "
                f"fills={self.n_fills}  sharpe={self.sharpe:.3f}")


class ASSimulator:

    def __init__(self, strategy: AvellanedaStoikov, generator,
                 T: float = 1.0, requote_dt: float = 0.05, lot_size: float = 5.0,
                 max_inventory: float = 50.0, record_every: int = 10, seed: int = 0):
        self.strategy      = strategy
        self.generator     = generator
        self.T             = T
        self.requote_dt    = requote_dt
        self.lot_size      = lot_size
        self.max_inventory = max_inventory
        self.record_every  = record_every
        self.seed          = seed

    def run(self) -> SimulationResult:
        from order_book import Side

        self.generator.reset(seed=self.seed)
        lob = self.generator.lob

        result       = SimulationResult()
        inventory    = 0.0
        cash         = 0.0
        bid_id: Optional[int] = None
        ask_id: Optional[int] = None
        last_requote = -self.requote_dt
        idx          = 0

        # Submit initial quotes before any event arrives
        S = self.generator.mid
        q = self.strategy.skewed_quote(S, inventory, 0.0)
        bid_id, _ = lob.submit_limit(Side.BID, q.bid, self.lot_size, 0.0)
        ask_id, _ = lob.submit_limit(Side.ASK, q.ask, self.lot_size, 0.0)
        last_requote = 0.0

        while True:
            event = self.generator.next_event(self.T)
            if event is None:
                break

            t = event.timestamp
            S = self.generator.mid

            bid_filled = ask_filled = 0.0
            for trade in event.trades:
                if trade.passive_id == bid_id:
                    inventory += trade.quantity; cash -= trade.price * trade.quantity
                    bid_filled += trade.quantity
                elif trade.passive_id == ask_id:
                    inventory -= trade.quantity; cash += trade.price * trade.quantity
                    ask_filled += trade.quantity

            # flatten if we blow through the hard limit
            if abs(inventory) > self.max_inventory:
                flat_side = Side.ASK if inventory > 0 else Side.BID
                _, flat_trades = lob.submit_market(flat_side, abs(inventory) * 0.5, t)
                for ft in flat_trades:
                    if flat_side == Side.ASK:
                        inventory -= ft.quantity; cash += ft.price * ft.quantity
                    else:
                        inventory += ft.quantity; cash -= ft.price * ft.quantity

            if t - last_requote >= self.requote_dt:
                # cancel old quotes and reprice
                if bid_id is not None: lob.cancel(bid_id)
                if ask_id is not None: lob.cancel(ask_id)
                q = self.strategy.skewed_quote(S, inventory, t)
                bid_id, _ = lob.submit_limit(Side.BID, q.bid, self.lot_size, t)
                ask_id, _ = lob.submit_limit(Side.ASK, q.ask, self.lot_size, t)
                last_requote = t
            else:
                q = self.strategy.quote(S, inventory, t)

            if idx % self.record_every == 0:
                result.steps.append(SimStep(
                    timestamp=t, mid=S, inventory=inventory, cash=cash,
                    pnl=cash + inventory * S, quote=q,
                    bid_filled=bid_filled, ask_filled=ask_filled))
            idx += 1

        result.finalise()
        return result
