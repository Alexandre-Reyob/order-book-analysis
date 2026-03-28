import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class Side(Enum):
    BID = "bid"
    ASK = "ask"

class OrderType(Enum):
    LIMIT  = "limit"
    MARKET = "market"

@dataclass
class Order:
    order_id:   int
    side:       Side
    order_type: OrderType
    price:      float
    quantity:   float
    timestamp:  float
    remaining:  float = field(init=False)

    def __post_init__(self):
        self.remaining = self.quantity

@dataclass
class Trade:
    timestamp:      float
    aggressor_id:   int
    passive_id:     int
    price:          float
    quantity:       float
    aggressor_side: Side


# price-time priority matching engine
# bids: max-heap (negated prices), asks: min-heap

class LimitOrderBook:

    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self._next_id  = 0
        self.timestamp = 0.0
        self._bids: Dict[float, deque] = defaultdict(deque)
        self._asks: Dict[float, deque] = defaultdict(deque)
        self._bid_heap: List[float] = []
        self._ask_heap: List[float] = []
        self._orders: Dict[int, Order] = {}
        self.trades: List[Trade] = []

    def submit_limit(self, side: Side, price: float, qty: float, timestamp: float) -> Tuple[int, List[Trade]]:
        price  = self._round(price)
        order  = self._new_order(side, OrderType.LIMIT, price, qty, timestamp)
        trades = self._match(order, timestamp)
        if order.remaining > 1e-9:
            self._insert(order)
        return order.order_id, trades

    def submit_market(self, side: Side, qty: float, timestamp: float) -> Tuple[int, List[Trade]]:
        price = float("inf") if side == Side.BID else 0.0  # BID paie n'importe quel prix, ASK accepte n'importe quoi
        order = self._new_order(side, OrderType.MARKET, price, qty, timestamp)
        return order.order_id, self._match(order, timestamp)

    def cancel(self, order_id: int) -> bool:
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        order.remaining = 0.0
        return True

    def best_bid(self) -> Optional[float]:
        return self._best(self._bid_heap, self._bids, negate=True)

    def best_ask(self) -> Optional[float]:
        return self._best(self._ask_heap, self._asks, negate=False)

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return (bb + ba) / 2 if bb is not None and ba is not None else None

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        return ba - bb if bb is not None and ba is not None else None

    def snapshot(self, depth: int = 10) -> Dict:
        bids = self._levels(self._bid_heap, self._bids, negate=True,  depth=depth)
        asks = self._levels(self._ask_heap, self._asks, negate=False, depth=depth)
        return {"timestamp": self.timestamp, "bids": bids, "asks": asks,
                "mid": self.mid_price(), "spread": self.spread()}

    def _new_order(self, side, otype, price, qty, ts) -> Order:
        self._next_id += 1
        self.timestamp = ts
        return Order(self._next_id, side, otype, price, qty, ts)

    def _round(self, p: float) -> float:
        return round(round(p / self.tick_size) * self.tick_size, 10)

    def _match(self, aggressor: Order, ts: float) -> List[Trade]:
        trades = []
        if aggressor.side == Side.BID:
            heap, levels, neg = self._ask_heap, self._asks, False
            can_trade = lambda ap, pp: ap >= pp
        else:
            heap, levels, neg = self._bid_heap, self._bids, True
            can_trade = lambda ap, pp: ap <= pp

        while aggressor.remaining > 1e-9 and heap:
            best = -heap[0] if neg else heap[0]
            if not can_trade(aggressor.price, best):
                break
            lvl = levels.get(best)
            if not lvl:
                heapq.heappop(heap); continue
            while lvl and aggressor.remaining > 1e-9:
                passive = lvl[0]
                if passive.remaining <= 1e-9:
                    lvl.popleft(); self._orders.pop(passive.order_id, None); continue
                qty = min(aggressor.remaining, passive.remaining)
                aggressor.remaining -= qty
                passive.remaining   -= qty
                t = Trade(ts, aggressor.order_id, passive.order_id, passive.price, qty, aggressor.side)
                trades.append(t); self.trades.append(t)
                if passive.remaining <= 1e-9:
                    lvl.popleft(); self._orders.pop(passive.order_id, None)
            if not lvl:
                levels.pop(best, None); heapq.heappop(heap)
        return trades

    def _insert(self, order: Order):
        self._orders[order.order_id] = order
        if order.side == Side.BID:
            if order.price not in self._bids or not self._bids[order.price]:
                heapq.heappush(self._bid_heap, -order.price)
            self._bids[order.price].append(order)
        else:
            if order.price not in self._asks or not self._asks[order.price]:
                heapq.heappush(self._ask_heap, order.price)
            self._asks[order.price].append(order)

    def _best(self, heap, levels, negate) -> Optional[float]:
        # lazy cleanup of stale heap entries -- TODO: could maintain a cleaner structure
        while heap:
            p = -heap[0] if negate else heap[0]
            lvl = levels.get(p)
            if lvl and any(o.remaining > 1e-9 for o in lvl):
                return p
            heapq.heappop(heap); levels.pop(p, None)
        return None

    def _levels(self, heap, levels, negate, depth) -> List[Tuple[float, float]]:
        result, tmp = [], list(heap)
        heapq.heapify(tmp)
        while tmp and len(result) < depth:
            raw = heapq.heappop(tmp)
            p   = -raw if negate else raw
            lvl = levels.get(p)
            if not lvl: continue
            qty = sum(o.remaining for o in lvl if o.remaining > 1e-9)
            if qty > 1e-9: result.append((p, qty))
        return result

    def __repr__(self) -> str:
        s = self.snapshot(depth=5)
        lines = ["── LOB ──────────────────────"]
        for p, q in reversed(s["asks"][:5]):
            lines.append(f"  ASK  {p:9.4f}  {q:8.2f}")
        lines.append(f"  mid  {s['mid']:.4f}  spread={s['spread']:.4f}" if s["mid"] else "  (empty)")
        for p, q in s["bids"][:5]:
            lines.append(f"  BID  {p:9.4f}  {q:8.2f}")
        return "\n".join(lines)


# 6-dim Hawkes: [limit_bid, limit_ask, mkt_buy, mkt_sell, cancel_bid, cancel_ask]

@dataclass
class LOBEvent:
    timestamp:  float
    event_type: str
    price:      Optional[float]
    quantity:   float
    order_id:   Optional[int]  = None
    trades:     List[Trade]    = field(default_factory=list)


class LOBGenerator:

    DIM   = 6
    NAMES = ["limit_bid", "limit_ask", "mkt_buy", "mkt_sell", "cancel_bid", "cancel_ask"]

    def __init__(self, mu=None, alpha=None, beta=None, mid0=100.0,
                 sigma_mid=0.02, tick_size=0.01, qty_mean=10.0, depth_scale=0.05, seed=42):
        self.rng = np.random.default_rng(seed)
        n = self.DIM

        if mu is None:
            mu = np.array([1.5, 1.5, 0.4, 0.4, 0.3, 0.3])
        if alpha is None:
            alpha = np.zeros((n, n))
            alpha[0,0]=0.30; alpha[0,2]=0.20
            alpha[2,0]=0.25; alpha[2,2]=0.35
            alpha[1,1]=0.30; alpha[1,3]=0.20
            alpha[3,1]=0.25; alpha[3,3]=0.35
            alpha[4,2]=0.15; alpha[4,4]=0.20
            alpha[5,3]=0.15; alpha[5,5]=0.20
        if beta is None:
            beta = np.full((n, n), 2.0)

        self.mu=np.asarray(mu,dtype=float); self.alpha=np.asarray(alpha,dtype=float)
        self.beta=np.asarray(beta,dtype=float)
        self.mid=mid0; self.sigma_mid=sigma_mid; self.tick_size=tick_size
        self.qty_mean=qty_mean; self.depth_scale=depth_scale
        self.lob = LimitOrderBook(tick_size=tick_size)
        self._seed_book()

    def generate(self, T: float) -> List[LOBEvent]:
        events = []
        t=0.0; R=np.zeros((self.DIM, self.DIM))
        lam=self.mu.copy(); lam_tot=lam.sum()

        while t < T:
            dt    = self.rng.exponential(1.0 / max(lam_tot, 1e-10))
            t_new = t + dt
            if t_new > T: break

            R_new       = R * np.exp(-self.beta * dt)
            lam_new     = np.maximum(self.mu + (self.alpha * R_new).sum(axis=1), 1e-10)
            lam_new_tot = lam_new.sum()

            if self.rng.uniform() <= lam_new_tot / max(lam_tot, 1e-10):
                dim = self.rng.choice(self.DIM, p=lam_new / lam_new_tot)
                R_new[:, dim] += self.alpha[:, dim]
                t = t_new
                self.mid += self.rng.normal(0, self.sigma_mid * np.sqrt(dt))
                ev = self._dispatch(dim, t)
                if ev is not None: events.append(ev)
            else:
                t = t_new

            R=R_new; lam=np.maximum(self.mu+(self.alpha*R).sum(axis=1),1e-10)
            lam_tot=lam.sum()

        return events

    def reset(self, mid0=None, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        if mid0 is not None: self.mid = mid0
        self.lob = LimitOrderBook(self.tick_size)
        self._seed_book()
        self._t       = 0.0
        self._R       = np.zeros((self.DIM, self.DIM))
        self._lam     = self.mu.copy()
        self._lam_tot = self._lam.sum()

    def next_event(self, T_max: float) -> Optional[LOBEvent]:
        """Advance simulation by one accepted event; returns None when t >= T_max."""
        while True:
            dt    = self.rng.exponential(1.0 / max(self._lam_tot, 1e-10))
            t_new = self._t + dt
            if t_new > T_max:
                return None

            R_new       = self._R * np.exp(-self.beta * dt)
            lam_new     = np.maximum(self.mu + (self.alpha * R_new).sum(axis=1), 1e-10)
            lam_new_tot = lam_new.sum()

            if self.rng.uniform() <= lam_new_tot / max(self._lam_tot, 1e-10):
                dim = self.rng.choice(self.DIM, p=lam_new / lam_new_tot)
                R_new[:, dim] += self.alpha[:, dim]
                self._t = t_new
                self.mid += self.rng.normal(0, self.sigma_mid * np.sqrt(dt))
                self._R       = R_new
                self._lam     = np.maximum(self.mu + (self.alpha * self._R).sum(axis=1), 1e-10)
                self._lam_tot = self._lam.sum()
                ev = self._dispatch(dim, self._t)
                if ev is not None:
                    return ev
            else:
                self._t       = t_new
                self._R       = R_new
                self._lam     = np.maximum(self.mu + (self.alpha * self._R).sum(axis=1), 1e-10)
                self._lam_tot = self._lam.sum()

    def _dispatch(self, dim: int, t: float) -> Optional[LOBEvent]:
        name = self.NAMES[dim]
        if name == "limit_bid":
            p   = max(self.mid - self.rng.exponential(self.depth_scale), self.tick_size)
            qty = max(self.rng.exponential(self.qty_mean), 1.0)
            oid, tr = self.lob.submit_limit(Side.BID, p, qty, t)
            return LOBEvent(t, name, p, qty, oid, tr)
        elif name == "limit_ask":
            p   = self.mid + self.rng.exponential(self.depth_scale)
            qty = max(self.rng.exponential(self.qty_mean), 1.0)
            oid, tr = self.lob.submit_limit(Side.ASK, p, qty, t)
            return LOBEvent(t, name, p, qty, oid, tr)
        elif name == "mkt_buy":
            qty = max(self.rng.exponential(self.qty_mean * 0.5), 1.0)
            oid, tr = self.lob.submit_market(Side.BID, qty, t)
            return LOBEvent(t, name, None, qty, oid, tr)
        elif name == "mkt_sell":
            qty = max(self.rng.exponential(self.qty_mean * 0.5), 1.0)
            oid, tr = self.lob.submit_market(Side.ASK, qty, t)
            return LOBEvent(t, name, None, qty, oid, tr)
        elif name in ("cancel_bid", "cancel_ask"):
            side  = Side.BID if name == "cancel_bid" else Side.ASK
            cands = [oid for oid, o in self.lob._orders.items()
                     if o.side == side and o.remaining > 1e-9]
            if not cands: return None
            target = int(self.rng.choice(cands))
            if self.lob.cancel(target):
                return LOBEvent(t, name, None, 0.0, target, [])
        return None

    def _seed_book(self):
        for i in range(1, 6):
            qty = max(self.rng.exponential(self.qty_mean) + 5, 1.0)
            self.lob.submit_limit(Side.BID, self.mid - i * self.depth_scale, qty, -0.001)
            self.lob.submit_limit(Side.ASK, self.mid + i * self.depth_scale, qty, -0.001)
