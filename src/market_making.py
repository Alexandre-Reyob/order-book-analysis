from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
        # NOT part of Avellaneda-Stoikov: an extra, ad hoc tilt near the inventory limit,
        # on top of the reservation-price skew. Only used when ASSimulator(extra_skew=True).
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
    quote:      Quote      # the quote actually resting in the book at that time
    bid_filled: float
    ask_filled: float


@dataclass
class Fill:
    timestamp: float
    side:      int        # +1 = market maker bought (bid hit), -1 = sold (ask lifted)
    price:     float
    quantity:  float


@dataclass
class SimulationResult:
    steps: List[SimStep] = field(default_factory=list)
    fills: List[Fill]    = field(default_factory=list)
    mid_times:  np.ndarray = field(default_factory=lambda: np.empty(0))
    mid_values: np.ndarray = field(default_factory=lambda: np.empty(0))

    final_pnl:     float = 0.0
    max_inventory: float = 0.0
    n_fills:       int   = 0      # passive fills (our resting quotes hit by market orders)
    n_crossed:     int   = 0      # executions of our own quotes that crossed the book on arrival

    def finalise(self):
        if not self.steps: return
        self.final_pnl     = float(self.steps[-1].pnl)
        self.max_inventory = float(max(abs(s.inventory) for s in self.steps))
        self.n_fills       = len(self.fills)
        # no Sharpe ratio here: PnL increments are per event, not per day, so annualising
        # with sqrt(252) is meaningless. Compare strategies on the distribution of final PnL
        # across many seeds instead (see monte_carlo).

    def mid_at(self, t: np.ndarray, before: bool = False) -> np.ndarray:
        # latent mid at time t (last value observed at or before t);
        # before=True -> last value strictly before t (i.e. just before the event at t)
        idx = np.searchsorted(self.mid_times, t, side="left" if before else "right") - 1
        return self.mid_values[np.clip(idx, 0, len(self.mid_values) - 1)]

    def markout(self, horizons: Sequence[float]) -> np.ndarray:
        """Volume-weighted average markout of passive fills at each horizon h:
        side * (mid(t+h) - fill price). h = 0 uses the mid just before the fill, i.e. the
        effective half-spread captured. Flat in h = no adverse selection;
        decreasing in h = fills are followed by adverse price moves."""
        if not self.fills:
            return np.full(len(horizons), np.nan)
        t  = np.array([f.timestamp for f in self.fills])
        sd = np.array([f.side      for f in self.fills])
        px = np.array([f.price     for f in self.fills])
        qt = np.array([f.quantity  for f in self.fills])
        out = []
        for h in horizons:
            ok = t + h <= self.mid_times[-1]
            if not ok.any():
                out.append(np.nan); continue
            mk = sd[ok] * (self.mid_at(t[ok] + h, before=(h == 0)) - px[ok])
            out.append(float(np.sum(mk * qt[ok]) / np.sum(qt[ok])))
        return np.array(out)

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
                f"fills={self.n_fills}  crossed={self.n_crossed}")


class ASSimulator:

    def __init__(self, strategy: AvellanedaStoikov, generator,
                 T: float = 1.0, requote_dt: float = 0.05, lot_size: float = 5.0,
                 max_inventory: float = 50.0, record_every: int = 10,
                 extra_skew: bool = False, seed: int = 0):
        self.strategy      = strategy
        self.generator     = generator
        self.T             = T
        self.requote_dt    = requote_dt
        self.lot_size      = lot_size
        self.max_inventory = max_inventory
        self.record_every  = record_every
        self.extra_skew    = extra_skew
        self.seed          = seed

    def _quote(self, S, q, t):
        if self.extra_skew:
            return self.strategy.skewed_quote(S, q, t)
        return self.strategy.quote(S, q, t)

    def run(self) -> SimulationResult:
        from order_book import Side

        self.generator.reset(seed=self.seed)
        lob = self.generator.lob

        result       = SimulationResult()
        inventory    = 0.0
        cash         = 0.0
        bid_id: Optional[int] = None
        ask_id: Optional[int] = None
        idx          = 0
        mid_t, mid_v = [0.0], [self.generator.mid]

        def post_quotes(quote, t):
            # A quote can cross the book (e.g. a large reservation-price shift puts the ask below
            # the best bid): it then executes immediately as an aggressive order. Those executions
            # must be booked too, otherwise inventory and cash are wrong.
            nonlocal inventory, cash
            b_id, b_trades = lob.submit_limit(Side.BID, quote.bid, self.lot_size, t)
            for tr in b_trades:
                inventory += tr.quantity; cash -= tr.price * tr.quantity
            a_id, a_trades = lob.submit_limit(Side.ASK, quote.ask, self.lot_size, t)
            for tr in a_trades:
                inventory -= tr.quantity; cash += tr.price * tr.quantity
            result.n_crossed += len(b_trades) + len(a_trades)
            return b_id, a_id

        # Submit initial quotes before any event arrives
        S = self.generator.mid
        live_quote = self._quote(S, inventory, 0.0)
        bid_id, ask_id = post_quotes(live_quote, 0.0)
        last_requote = 0.0

        while True:
            event = self.generator.next_event(self.T)
            if event is None:
                break

            t = event.timestamp
            S = self.generator.mid
            mid_t.append(t); mid_v.append(S)

            bid_filled = ask_filled = 0.0
            for trade in event.trades:
                if trade.passive_id == bid_id:
                    inventory += trade.quantity; cash -= trade.price * trade.quantity
                    bid_filled += trade.quantity
                    result.fills.append(Fill(t, +1, trade.price, trade.quantity))
                elif trade.passive_id == ask_id:
                    inventory -= trade.quantity; cash += trade.price * trade.quantity
                    ask_filled += trade.quantity
                    result.fills.append(Fill(t, -1, trade.price, trade.quantity))

            # flatten if we blow through the hard limit
            if abs(inventory) > self.max_inventory:
                # pull our own quotes first so the market order cannot trade against them
                if bid_id is not None: lob.cancel(bid_id)
                if ask_id is not None: lob.cancel(ask_id)
                last_requote = -np.inf
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
                live_quote = self._quote(S, inventory, t)
                bid_id, ask_id = post_quotes(live_quote, t)
                last_requote = t

            if idx % self.record_every == 0:
                # record the quote actually resting in the book, not a recomputed one
                result.steps.append(SimStep(
                    timestamp=t, mid=S, inventory=inventory, cash=cash,
                    pnl=cash + inventory * S, quote=live_quote,
                    bid_filled=bid_filled, ask_filled=ask_filled))
            idx += 1

        result.mid_times  = np.array(mid_t)
        result.mid_values = np.array(mid_v)
        result.finalise()
        return result


# ── calibration of the strategy to the simulated market ─────────────────────

def simulate_flow(generator, T: float, seed: int = 0):
    """Run the generator alone (no market maker). Returns the latent mid path and the events."""
    generator.reset(seed=seed)
    times, mids, events = [0.0], [generator.mid], []
    while True:
        ev = generator.next_event(T)
        if ev is None: break
        times.append(ev.timestamp); mids.append(generator.mid); events.append(ev)
    return np.array(times), np.array(mids), events


def estimate_sigma(times: np.ndarray, mids: np.ndarray) -> float:
    """sigma of dS = sigma dW from an irregularly sampled path: sqrt(sum dS^2 / sum dt).
    Includes impact moves if the generator has impact > 0."""
    dS, dt = np.diff(mids), np.diff(times)
    return float(np.sqrt(np.sum(dS**2) / np.sum(dt)))


def estimate_fill_intensity(events, T: float, deltas: Sequence[float],
                            min_count: int = 20) -> Tuple[float, float, np.ndarray]:
    """Fit lambda(delta) = A * exp(-kappa * delta), the rate at which a quote resting at distance
    delta from the mid would be reached by market orders. For each market order we measure how
    far it walked the book (worst fill price vs the latent mid just before it), then count, per side,
    the orders that reached at least delta. Approximation: ignores the queue at that price level
    and the fact that our own quote would change the book. The log-linear fit only uses deltas
    reached by at least min_count orders, so that a handful of very deep orders do not drive it.
    Returns (A, kappa, lambda_hat per delta)."""
    reach = []
    for ev in events:
        if ev.event_type not in ("mkt_buy", "mkt_sell") or not ev.trades or ev.mid is None:
            continue
        if ev.event_type == "mkt_buy":
            reach.append(max(tr.price for tr in ev.trades) - ev.mid)
        else:
            reach.append(ev.mid - min(tr.price for tr in ev.trades))
    reach = np.array(reach)
    deltas = np.asarray(deltas, dtype=float)
    # divide by 2T: both sides pooled, intensity per side
    counts = np.array([(reach >= d).sum() for d in deltas])
    lam = counts / (2.0 * T)
    ok = counts >= min_count
    slope, intercept = np.polyfit(deltas[ok], np.log(lam[ok]), 1)
    return float(np.exp(intercept)), float(-slope), lam


def monte_carlo(make_strategy: Callable[[], AvellanedaStoikov], make_generator: Callable[[], object],
                seeds: Sequence[int], horizons: Sequence[float] = (), **sim_kwargs) -> Dict[str, np.ndarray]:
    """Run the same setup over many seeds; returns per-seed final PnL, max |inventory|,
    number of fills and markouts at the given horizons."""
    pnl, inv, nf, nc, mk = [], [], [], [], []
    for s in seeds:
        res = ASSimulator(make_strategy(), make_generator(), seed=s, **sim_kwargs).run()
        pnl.append(res.final_pnl); inv.append(res.max_inventory)
        nf.append(res.n_fills); nc.append(res.n_crossed)
        if len(horizons): mk.append(res.markout(horizons))
    out = {"pnl": np.array(pnl), "max_inv": np.array(inv),
           "fills": np.array(nf), "crossed": np.array(nc)}
    if len(horizons): out["markout"] = np.array(mk)
    return out
