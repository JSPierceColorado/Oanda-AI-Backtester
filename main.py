import os
import json
import time
import math
import base64
import random
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Logging
# ============================================================
def setup_logging() -> None:
    level = (os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


log = logging.getLogger("evolver")


# ============================================================
# Env helpers
# ============================================================
def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return False
    s = str(x).strip().lower()
    return s in ("true", "1", "yes", "y", "t", "tradeable")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def now_utc_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def max_drawdown(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return 0.0
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - (equity / peak)
    return float(np.max(dd))


# ============================================================
# Expression engine (random strategy programs)
# ============================================================
# Node formats (dict):
#  - {"t":"col","name":"dist_sma50_pct"}
#  - {"t":"const","v":0.123}
#  - {"t":"u","op":"abs|neg|tanh|log1p|sqrt|clip","a":node,"lo":-1,"hi":1}
#  - {"t":"b","op":"add|sub|mul|div|max|min|pow","a":node,"b":node}
#  - {"t":"cmp","op":"gt|lt|ge|le","a":node,"b":node} -> bool
#  - {"t":"log","op":"and|or","a":boolnode,"b":boolnode} -> bool
#  - {"t":"not","a":boolnode} -> bool
#  - {"t":"if","cond":boolnode,"x":node,"y":node}
#
# Evaluation is vectorized; unknown columns -> NaN arrays (safe).
UNARY_OPS = ("abs", "neg", "tanh", "log1p", "sqrt", "clip")
BINARY_OPS = ("add", "sub", "mul", "div", "max", "min", "pow")
CMP_OPS = ("gt", "lt", "ge", "le")
LOG_OPS = ("and", "or")


def _nan_array(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def _safe_pow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # pow can easily overflow; keep it safe-ish
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.power(a, b)
    out[~np.isfinite(out)] = np.nan
    return out


def eval_num(node: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    t = node.get("t")
    if t == "col":
        name = node.get("name", "")
        if name in df.columns:
            arr = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            arr = np.array(arr, dtype=float, copy=True)
            arr[~np.isfinite(arr)] = np.nan
            return arr
        return _nan_array(n)

    if t == "const":
        v = float(node.get("v", 0.0))
        return np.full(n, v, dtype=float)

    if t == "u":
        op = node.get("op")
        a = eval_num(node.get("a", {"t": "const", "v": 0.0}), df)
        if op == "abs":
            return np.abs(a)
        if op == "neg":
            return -a
        if op == "tanh":
            return np.tanh(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0))
        if op == "log1p":
            # log1p(abs(x)) preserves domain
            return np.log1p(np.abs(a))
        if op == "sqrt":
            with np.errstate(invalid="ignore"):
                out = np.sqrt(np.abs(a))
            out[~np.isfinite(out)] = np.nan
            return out
        if op == "clip":
            lo = float(node.get("lo", -1.0))
            hi = float(node.get("hi", 1.0))
            return np.clip(a, lo, hi)
        return a

    if t == "b":
        op = node.get("op")
        a = eval_num(node.get("a", {"t": "const", "v": 0.0}), df)
        b = eval_num(node.get("b", {"t": "const", "v": 1.0}), df)
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "div":
            return _safe_div(a, b)
        if op == "max":
            return np.fmax(a, b)
        if op == "min":
            return np.fmin(a, b)
        if op == "pow":
            return _safe_pow(a, b)
        return a

    if t == "if":
        cond = eval_bool(node.get("cond", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), df)
        x = eval_num(node.get("x", {"t": "const", "v": 1.0}), df)
        y = eval_num(node.get("y", {"t": "const", "v": -1.0}), df)
        out = np.where(cond, x, y)
        out = np.array(out, dtype=float, copy=True)
        out[~np.isfinite(out)] = np.nan
        return out

    return _nan_array(n)


def eval_bool(node: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    t = node.get("t")
    if t == "cmp":
        op = node.get("op")
        a = eval_num(node.get("a", {"t": "const", "v": 0.0}), df)
        b = eval_num(node.get("b", {"t": "const", "v": 0.0}), df)
        if op == "gt":
            return np.greater(a, b, where=~np.isnan(a) & ~np.isnan(b), out=np.zeros(n, dtype=bool))
        if op == "lt":
            return np.less(a, b, where=~np.isnan(a) & ~np.isnan(b), out=np.zeros(n, dtype=bool))
        if op == "ge":
            return np.greater_equal(a, b, where=~np.isnan(a) & ~np.isnan(b), out=np.zeros(n, dtype=bool))
        if op == "le":
            return np.less_equal(a, b, where=~np.isnan(a) & ~np.isnan(b), out=np.zeros(n, dtype=bool))
        return np.zeros(n, dtype=bool)

    if t == "log":
        op = node.get("op")
        a = eval_bool(node.get("a"), df)
        b = eval_bool(node.get("b"), df)
        if op == "and":
            return a & b
        if op == "or":
            return a | b
        return a

    if t == "not":
        a = eval_bool(node.get("a"), df)
        return ~a

    # fall back: treat numeric node as bool by > 0
    arr = eval_num(node, df)
    return np.isfinite(arr) & (arr > 0)


def expr_to_str(node: Dict[str, Any]) -> str:
    """Human-ish compact representation for debugging (not perfect)."""
    t = node.get("t")
    if t == "col":
        return str(node.get("name"))
    if t == "const":
        return f"{float(node.get('v', 0.0)):.4g}"
    if t == "u":
        op = node.get("op")
        a = expr_to_str(node.get("a", {}))
        if op == "clip":
            return f"clip({a},{node.get('lo',-1)},{node.get('hi',1)})"
        return f"{op}({a})"
    if t == "b":
        op = node.get("op")
        a = expr_to_str(node.get("a", {}))
        b = expr_to_str(node.get("b", {}))
        return f"({a} {op} {b})"
    if t == "cmp":
        return f"({expr_to_str(node.get('a', {}))} {node.get('op')} {expr_to_str(node.get('b', {}))})"
    if t == "log":
        return f"({expr_to_str(node.get('a', {}))} {node.get('op')} {expr_to_str(node.get('b', {}))})"
    if t == "not":
        return f"not({expr_to_str(node.get('a', {}))})"
    if t == "if":
        return f"if({expr_to_str(node.get('cond', {}))},{expr_to_str(node.get('x', {}))},{expr_to_str(node.get('y', {}))})"
    return "nan"


def random_const(rng: random.Random) -> float:
    # Wide distribution: small numbers + occasional big ones
    if rng.random() < 0.85:
        return rng.gauss(0, 1.0)
    return rng.gauss(0, 5.0)


def random_num_expr(rng: random.Random, cols: List[str], depth: int) -> Dict[str, Any]:
    if depth <= 0 or rng.random() < 0.35:
        if cols and rng.random() < 0.7:
            return {"t": "col", "name": rng.choice(cols)}
        return {"t": "const", "v": random_const(rng)}

    # occasionally build an if-then-else
    if rng.random() < 0.10 and depth >= 1:
        return {
            "t": "if",
            "cond": random_bool_expr(rng, cols, depth - 1),
            "x": random_num_expr(rng, cols, depth - 1),
            "y": random_num_expr(rng, cols, depth - 1),
        }

    if rng.random() < 0.40:
        op = rng.choice(UNARY_OPS)
        node = {"t": "u", "op": op, "a": random_num_expr(rng, cols, depth - 1)}
        if op == "clip":
            node["lo"] = random_const(rng)
            node["hi"] = node["lo"] + abs(random_const(rng)) + 0.5
        return node

    op = rng.choice(BINARY_OPS)
    return {
        "t": "b",
        "op": op,
        "a": random_num_expr(rng, cols, depth - 1),
        "b": random_num_expr(rng, cols, depth - 1),
    }


def random_bool_expr(rng: random.Random, cols: List[str], depth: int) -> Dict[str, Any]:
    if depth <= 0 or rng.random() < 0.55:
        return {
            "t": "cmp",
            "op": rng.choice(CMP_OPS),
            "a": random_num_expr(rng, cols, 0),
            "b": random_num_expr(rng, cols, 0),
        }
    if rng.random() < 0.15:
        return {"t": "not", "a": random_bool_expr(rng, cols, depth - 1)}
    return {
        "t": "log",
        "op": rng.choice(LOG_OPS),
        "a": random_bool_expr(rng, cols, depth - 1),
        "b": random_bool_expr(rng, cols, depth - 1),
    }


def mutate_expr_any(node: Dict[str, Any], rng: random.Random, cols: List[str], max_depth: int) -> Dict[str, Any]:
    """Mutation capable of *anything*: tweak constants, swap features, change ops, replace subtrees, wrap/compose."""
    # occasionally nuke the whole subtree
    if rng.random() < 0.10:
        return random_num_expr(rng, cols, max_depth)

    t = node.get("t")

    # Replace leaf
    if t in ("col", "const") and rng.random() < 0.35:
        return random_num_expr(rng, cols, 1)

    out = dict(node)

    # Mutate different node types
    if t == "col":
        if cols and rng.random() < 0.75:
            out["name"] = rng.choice(cols)
        return out

    if t == "const":
        v = float(out.get("v", 0.0))
        if rng.random() < 0.75:
            v += rng.gauss(0, 0.5)
        else:
            v = random_const(rng)
        out["v"] = float(v)
        return out

    if t == "u":
        if rng.random() < 0.30:
            out["op"] = rng.choice(UNARY_OPS)
        out["a"] = mutate_expr_any(out.get("a", {"t": "const", "v": 0.0}), rng, cols, max(0, max_depth - 1))
        if out.get("op") == "clip":
            if rng.random() < 0.5:
                out["lo"] = random_const(rng)
            if rng.random() < 0.5:
                out["hi"] = float(out.get("lo", -1.0)) + abs(random_const(rng)) + 0.5
        # sometimes wrap again
        if rng.random() < 0.15:
            out = {"t": "u", "op": rng.choice(UNARY_OPS), "a": out}
        return out

    if t == "b":
        if rng.random() < 0.30:
            out["op"] = rng.choice(BINARY_OPS)
        out["a"] = mutate_expr_any(out.get("a", {"t": "const", "v": 0.0}), rng, cols, max(0, max_depth - 1))
        out["b"] = mutate_expr_any(out.get("b", {"t": "const", "v": 1.0}), rng, cols, max(0, max_depth - 1))
        # sometimes compose with a new random subtree
        if rng.random() < 0.15:
            out = {"t": "b", "op": rng.choice(BINARY_OPS), "a": out, "b": random_num_expr(rng, cols, 2)}
        return out

    if t == "if":
        out["cond"] = mutate_bool_any(out.get("cond", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), rng, cols, max(0, max_depth - 1))
        out["x"] = mutate_expr_any(out.get("x", {"t": "const", "v": 1.0}), rng, cols, max(0, max_depth - 1))
        out["y"] = mutate_expr_any(out.get("y", {"t": "const", "v": -1.0}), rng, cols, max(0, max_depth - 1))
        return out

    # fallback: replace
    return random_num_expr(rng, cols, max_depth)


def mutate_bool_any(node: Dict[str, Any], rng: random.Random, cols: List[str], max_depth: int) -> Dict[str, Any]:
    if rng.random() < 0.10:
        return random_bool_expr(rng, cols, max_depth)

    t = node.get("t")
    out = dict(node)

    if t == "cmp":
        if rng.random() < 0.35:
            out["op"] = rng.choice(CMP_OPS)
        out["a"] = mutate_expr_any(out.get("a", {"t": "const", "v": 0.0}), rng, cols, 2)
        out["b"] = mutate_expr_any(out.get("b", {"t": "const", "v": 0.0}), rng, cols, 2)
        return out

    if t == "log":
        if rng.random() < 0.30:
            out["op"] = rng.choice(LOG_OPS)
        out["a"] = mutate_bool_any(out.get("a", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), rng, cols, max(0, max_depth - 1))
        out["b"] = mutate_bool_any(out.get("b", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), rng, cols, max(0, max_depth - 1))
        # sometimes add another clause
        if rng.random() < 0.15:
            out = {"t": "log", "op": rng.choice(LOG_OPS), "a": out, "b": random_bool_expr(rng, cols, 1)}
        return out

    if t == "not":
        out["a"] = mutate_bool_any(out.get("a", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), rng, cols, max(0, max_depth - 1))
        return out

    # fallback
    return random_bool_expr(rng, cols, max_depth)


# ============================================================
# Strategy genome (expanded)
# ============================================================
@dataclass
class Genome:
    # The "program" producing a numeric score
    score_expr: Dict[str, Any]

    # Optional filter conditions (bool programs). All must pass.
    filters: List[Dict[str, Any]]

    # How to turn scores into trades:
    # - "threshold": long if score > thr, short if score < -thr
    # - "topk": per timestamp pick top_k longs and top_k shorts by score
    # - "random": ignore score, pick random trades per timestamp
    select_mode: str  # threshold|topk|random

    thr: float
    top_k: int
    side_mode: str  # both|long_only|short_only

    horizon: int

    cost_mult: float  # multiplier on spread_bps/10000 cost
    random_trade_prob: float  # for random mode: probability per row before top_k truncation

    def normalize(self) -> "Genome":
        self.select_mode = self.select_mode if self.select_mode in ("threshold", "topk", "random") else "threshold"
        self.side_mode = self.side_mode if self.side_mode in ("both", "long_only", "short_only") else "both"
        self.thr = clamp(float(self.thr), 0.0, 100.0)
        self.top_k = int(clamp(int(self.top_k), 1, 200))
        self.horizon = int(clamp(int(self.horizon), 1, 200))
        self.cost_mult = clamp(float(self.cost_mult), 0.0, 50.0)
        self.random_trade_prob = clamp(float(self.random_trade_prob), 0.0, 1.0)
        if self.filters is None:
            self.filters = []
        return self


def genome_id(g: Genome) -> str:
    payload = json.dumps(asdict(g), sort_keys=True)
    return str(abs(hash(payload)))[:10]


def random_genome(rng: random.Random, cols: List[str]) -> Genome:
    # occasionally generate a fully random strategy mode (no sense required)
    select_mode = rng.choices(["threshold", "topk", "random"], weights=[0.55, 0.35, 0.10], k=1)[0]

    filters: List[Dict[str, Any]] = []
    # 0..4 random filters
    for _ in range(rng.randint(0, 4)):
        filters.append(random_bool_expr(rng, cols, depth=rng.randint(0, 2)))

    g = Genome(
        score_expr=random_num_expr(rng, cols, depth=rng.randint(1, 4)),
        filters=filters,
        select_mode=select_mode,
        thr=abs(random_const(rng)),
        top_k=rng.choice([1, 2, 3, 5, 8, 13]),
        side_mode=rng.choice(["both", "long_only", "short_only"]),
        horizon=rng.choice([1, 2, 3, 6, 12]),
        cost_mult=abs(rng.gauss(1.0, 0.4)),
        random_trade_prob=clamp(rng.random() * 0.35, 0.01, 0.35),
    )
    return g.normalize()


def mutate(g: Genome, rng: random.Random, cols: List[str], max_expr_depth: int) -> Genome:
    """Mutations capable of any/every modification."""
    if rng.random() < 0.06:
        # full random restart
        return random_genome(rng, cols)

    h = Genome(**asdict(g)).normalize()

    # mutate score expression hard
    if rng.random() < 0.85:
        h.score_expr = mutate_expr_any(h.score_expr, rng, cols, max_expr_depth)

    # mutate filters
    if rng.random() < 0.65:
        # add/remove/modify filters
        if rng.random() < 0.35 and len(h.filters) < 10:
            h.filters.append(random_bool_expr(rng, cols, depth=rng.randint(0, 2)))
        if rng.random() < 0.25 and len(h.filters) > 0:
            h.filters.pop(rng.randrange(len(h.filters)))
        # mutate existing
        for i in range(len(h.filters)):
            if rng.random() < 0.60:
                h.filters[i] = mutate_bool_any(h.filters[i], rng, cols, max_expr_depth)

    # mutate selection mode + params
    if rng.random() < 0.25:
        h.select_mode = rng.choice(["threshold", "topk", "random"])
    if rng.random() < 0.50:
        h.thr = abs(h.thr + rng.gauss(0, 0.75))
    if rng.random() < 0.35:
        h.top_k = int(clamp(h.top_k + rng.choice([-5, -3, -1, 1, 3, 5]), 1, 200))
    if rng.random() < 0.15:
        h.side_mode = rng.choice(["both", "long_only", "short_only"])

    if rng.random() < 0.40:
        h.horizon = rng.choice([1, 2, 3, 4, 6, 8, 12, 16])

    if rng.random() < 0.45:
        h.cost_mult = abs(h.cost_mult + rng.gauss(0, 0.35))

    if rng.random() < 0.35:
        h.random_trade_prob = clamp(h.random_trade_prob + rng.gauss(0, 0.08), 0.0, 1.0)

    return h.normalize()


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    da = asdict(a)
    db = asdict(b)
    child = {}
    for k in da.keys():
        child[k] = da[k] if rng.random() < 0.5 else db[k]
    return Genome(**child).normalize()


# ============================================================
# Data sources (Google Sheets OR Excel)
# ============================================================
class SheetClient:
    def read_screener(self) -> pd.DataFrame:
        raise NotImplementedError

    def read_history(self, lookback_rows: int) -> pd.DataFrame:
        raise NotImplementedError

    def append_strategies(self, header: List[str], rows: List[List[Any]]) -> None:
        raise NotImplementedError

    def write_champion(self, header: List[str], row: List[Any]) -> None:
        raise NotImplementedError

    def replace_signals(self, header: List[str], rows: List[List[Any]]) -> None:
        raise NotImplementedError


class ExcelClient(SheetClient):
    """Local testing helper; prints outputs."""
    def __init__(self, path: str, screener_tab: str, history_tab: str):
        self.path = path
        self.screener_tab = screener_tab
        self.history_tab = history_tab

    def read_screener(self) -> pd.DataFrame:
        return pd.read_excel(self.path, sheet_name=self.screener_tab)

    def read_history(self, lookback_rows: int) -> pd.DataFrame:
        hist_raw = pd.read_excel(self.path, sheet_name=self.history_tab, header=None)
        scr = self.read_screener()
        hist_raw = hist_raw.tail(lookback_rows) if lookback_rows > 0 else hist_raw
        hist_raw.columns = list(scr.columns)
        return hist_raw

    def append_strategies(self, header: List[str], rows: List[List[Any]]) -> None:
        log.info("STRATEGIES_HEADER %s", header)
        for r in rows:
            log.info("STRATEGY_ROW %s", r)

    def write_champion(self, header: List[str], row: List[Any]) -> None:
        log.info("CHAMPION_HEADER %s", header)
        log.info("CHAMPION_ROW %s", row)

    def replace_signals(self, header: List[str], rows: List[List[Any]]) -> None:
        log.info("SIGNALS_HEADER %s", header)
        for r in rows:
            log.info("SIGNAL_ROW %s", r)


class GoogleSheetsClient(SheetClient):
    def __init__(
        self,
        sheet_id: str,
        screener_tab: str,
        history_tab: str,
        strategies_tab: str,
        signals_tab: str,
        champion_tab: str,
    ):
        import gspread
        from google.oauth2 import service_account

        sa_json = env_str("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not sa_json:
            raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON env var.")

        # raw JSON or base64 JSON
        try:
            if sa_json.strip().startswith("{"):
                info = json.loads(sa_json)
            else:
                info = json.loads(base64.b64decode(sa_json).decode("utf-8"))
        except Exception as e:
            raise RuntimeError("Failed to parse GOOGLE_SERVICE_ACCOUNT_JSON (raw json or base64).") from e

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(creds)
        self.sh = gc.open_by_key(sheet_id)

        self.screener_tab = screener_tab
        self.history_tab = history_tab
        self.strategies_tab = strategies_tab
        self.signals_tab = signals_tab
        self.champion_tab = champion_tab

    def _with_retry(self, fn, *, what: str, max_tries: int = 6):
        """Retry wrapper mainly for Google Sheets 429 quota errors."""
        from gspread.exceptions import APIError

        delay = 1.0
        for attempt in range(1, max_tries + 1):
            try:
                return fn()
            except APIError as e:
                msg = str(e)
                if "429" in msg or "Quota exceeded" in msg:
                    log.warning("Sheets quota hit during %s (attempt %s/%s). Sleeping %.1fs", what, attempt, max_tries, delay)
                    time.sleep(delay)
                    delay = min(delay * 2.0, 30.0)
                    continue
                raise
        raise RuntimeError(f"Exceeded retry budget for {what}")

    def _ws(self, name: str, rows: int = 1000, cols: int = 40):
        import gspread

        try:
            return self.sh.worksheet(name)
        except gspread.WorksheetNotFound:
            return self.sh.add_worksheet(title=name, rows=str(rows), cols=str(cols))

    def read_screener(self) -> pd.DataFrame:
        ws = self._ws(self.screener_tab, rows=2000, cols=200)
        values = self._with_retry(lambda: ws.get_all_values(), what=f"read {ws.title}")
        if not values:
            return pd.DataFrame()
        header = values[0]
        rows = values[1:]
        return pd.DataFrame(rows, columns=header)

    def read_history(self, lookback_rows: int) -> pd.DataFrame:
        ws = self._ws(self.history_tab, rows=50000, cols=200)
        values = self._with_retry(lambda: ws.get_all_values(), what=f"read {ws.title}")
        if not values:
            return pd.DataFrame()

        # Detect whether history has headers
        if values[0] and str(values[0][0]).strip().lower() == "symbol":
            header = values[0]
            data_rows = values[1:]
        else:
            scr = self.read_screener()
            header = list(scr.columns) if not scr.empty else [f"col_{i}" for i in range(len(values[0]))]
            data_rows = values

        def is_blank(row):
            return all((c is None) or (str(c).strip() == "") for c in row)

        data_rows = [r for r in data_rows if not is_blank(r)]
        if lookback_rows > 0:
            data_rows = data_rows[-lookback_rows:]

        return pd.DataFrame(data_rows, columns=header)

    def _ensure_header_row(self, ws, header: List[str]) -> None:
        values = self._with_retry(lambda: ws.get_all_values(), what=f"read {ws.title}")
        if not values:
            self._with_retry(lambda: ws.append_row(header, value_input_option="RAW"), what=f"append_row {ws.title}")
            return
        # If first row isn't our header, prepend it (common when sheet existed but was blank/formatted)
        first = values[0]
        if len(first) < len(header) or any(str(first[i]).strip() != header[i] for i in range(min(len(first), len(header)))):
            self._with_retry(lambda: ws.insert_row(header, 1, value_input_option="RAW"), what=f"insert_row {ws.title}")

    def append_strategies(self, header: List[str], rows: List[List[Any]]) -> None:
        ws = self._ws(self.strategies_tab, rows=20000, cols=max(40, len(header) + 2))
        self._ensure_header_row(ws, header)
        self._with_retry(lambda: ws.append_rows(rows, value_input_option="RAW"), what=f"append_rows {ws.title}")

    def write_champion(self, header: List[str], row: List[Any]) -> None:
        ws = self._ws(self.champion_tab, rows=2000, cols=max(40, len(header) + 2))
        self._with_retry(lambda: ws.clear(), what=f"clear {ws.title}")
        self._with_retry(lambda: ws.append_row(header, value_input_option="RAW"), what=f"append_row {ws.title}")
        self._with_retry(lambda: ws.append_row(row, value_input_option="RAW"), what=f"append_row {ws.title}")

    def replace_signals(self, header: List[str], rows: List[List[Any]]) -> None:
        ws = self._ws(self.signals_tab, rows=2000, cols=max(40, len(header) + 2))
        self._with_retry(lambda: ws.clear(), what=f"clear {ws.title}")
        self._with_retry(lambda: ws.append_row(header, value_input_option="RAW"), what=f"append_row {ws.title}")
        if rows:
            self._with_retry(lambda: ws.append_rows(rows, value_input_option="RAW"), what=f"append_rows {ws.title}")


# ============================================================
# Backtest
# ============================================================
KEY_COLS = {"symbol", "asof_utc"}


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Drop blank separator rows
    if "symbol" in d.columns:
        d["symbol"] = d["symbol"].astype(str)
        d = d[d["symbol"].notna() & (d["symbol"].str.strip() != "")].copy()

    if "asof_utc" in d.columns:
        d["asof_utc"] = pd.to_datetime(d["asof_utc"], errors="coerce", utc=True)
        d = d[d["asof_utc"].notna()].copy()

    # Convert everything except obvious string columns into numeric when possible
    for c in d.columns:
        if c in ("symbol", "asof_utc", "status"):
            continue
        # Keep tradeable as is; handled later
        if c == "tradeable":
            continue
        # Best-effort numeric conversion; non-numeric becomes NaN (safe for expr)
        d[c] = pd.to_numeric(d[c], errors="coerce")

    return d


def add_forward_returns(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    if "close" not in df.columns:
        raise RuntimeError("history is missing 'close' column; can't compute returns.")
    d = df.sort_values(["symbol", "asof_utc"]).reset_index(drop=True)
    for h in horizons:
        d[f"fwd_ret_{h}"] = d.groupby("symbol")["close"].shift(-h) / d["close"] - 1.0
    return d


def apply_filters(d: pd.DataFrame, filters: List[Dict[str, Any]]) -> np.ndarray:
    if not filters:
        return np.ones(len(d), dtype=bool)
    ok = np.ones(len(d), dtype=bool)
    for f in filters:
        try:
            ok = ok & eval_bool(f, d)
        except Exception:
            ok = ok & np.zeros(len(d), dtype=bool)
    return ok


def make_signals_for_frame(d: pd.DataFrame, g: Genome, rng_seed: int) -> pd.Series:
    """
    Returns a Series of signals indexed like d:
      +1 long, -1 short, 0 none.
    """
    n = len(d)
    sig = np.zeros(n, dtype=int)

    # Optional "tradeable" or "status" filters (if present)
    if "tradeable" in d.columns:
        tradeable_mask = d["tradeable"].map(to_bool).to_numpy(dtype=bool)
    else:
        tradeable_mask = np.ones(n, dtype=bool)

    if "status" in d.columns:
        status_mask = (d["status"].astype(str).str.lower() == "tradeable").to_numpy(dtype=bool)
    else:
        status_mask = np.ones(n, dtype=bool)

    base_ok = tradeable_mask & status_mask

    filt_ok = apply_filters(d, g.filters)
    ok = base_ok & filt_ok

    if not np.any(ok):
        return pd.Series(sig, index=d.index)

    # Select mode
    if g.select_mode == "random":
        rng = np.random.default_rng(rng_seed)
        chosen = rng.random(n) < g.random_trade_prob
        chosen = chosen & ok

        # random side
        side = rng.choice([-1, 1], size=n)
        if g.side_mode == "long_only":
            side = np.ones(n, dtype=int)
        elif g.side_mode == "short_only":
            side = -np.ones(n, dtype=int)

        sig = np.where(chosen, side, 0)
        return pd.Series(sig, index=d.index)

    # score-based
    score = eval_num(g.score_expr, d)
    score = np.array(score, dtype=float, copy=True)
    score[~np.isfinite(score)] = np.nan

    if g.select_mode == "threshold":
        long = ok & np.isfinite(score) & (score > g.thr)
        short = ok & np.isfinite(score) & (score < -g.thr)

        if g.side_mode == "long_only":
            short[:] = False
        elif g.side_mode == "short_only":
            long[:] = False

        sig = np.where(long, 1, np.where(short, -1, 0))
        return pd.Series(sig, index=d.index)

    # topk: per timestamp
    if "asof_utc" not in d.columns:
        return pd.Series(sig, index=d.index)

    tmp = d[["asof_utc"]].copy()
    tmp["score"] = score
    tmp["ok"] = ok
    tmp["idx"] = np.arange(len(tmp))

    # We'll mark signals in a vector, then map back.
    sig = np.zeros(len(tmp), dtype=int)

    # groupby timestamp
    for ts, grp in tmp.groupby("asof_utc", sort=False):
        gg = grp[grp["ok"] & np.isfinite(grp["score"])]

        if gg.empty:
            continue

        # longs: highest scores
        if g.side_mode != "short_only":
            longs = gg.sort_values("score", ascending=False).head(g.top_k)
            sig[longs["idx"].to_numpy(dtype=int)] = 1

        # shorts: lowest scores
        if g.side_mode != "long_only":
            shorts = gg.sort_values("score", ascending=True).head(g.top_k)
            sig[shorts["idx"].to_numpy(dtype=int)] = -1

    return pd.Series(sig, index=d.index)


def trade_returns(df: pd.DataFrame, g: Genome, rng_seed: int) -> pd.DataFrame:
    h = g.horizon
    col = f"fwd_ret_{h}"
    if col not in df.columns:
        return df.iloc[0:0].copy()

    d = df.copy()

    # Require forward return available and numeric
    d = d[d[col].notna()].copy()
    if d.empty:
        return d

    sig = make_signals_for_frame(d, g, rng_seed=rng_seed).to_numpy(dtype=int)
    d["signal"] = sig
    d = d[d["signal"] != 0].copy()
    if d.empty:
        return d

    # cost model
    spread = d["spread_bps"] if "spread_bps" in d.columns else 0.0
    spread = pd.to_numeric(spread, errors="coerce").fillna(0.0)
    cost = g.cost_mult * (spread / 10000.0)

    ret = d["signal"].astype(float) * d[col].astype(float) - cost.astype(float)
    out = d[["asof_utc", "symbol"]].copy()
    out["ret"] = ret.to_numpy(dtype=float)
    out["signal"] = d["signal"].to_numpy(dtype=int)
    return out


def score_rets(rets: np.ndarray, min_trades: int) -> Dict[str, Any]:
    n = int(len(rets))
    if n == 0:
        return {"n": 0, "fitness": -1e12, "total": 0.0, "sharpe": 0.0, "mdd": 0.0, "mean": 0.0, "std": 0.0}

    mean = float(np.mean(rets))
    std = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    sharpe = 0.0 if std == 0.0 else mean / std * math.sqrt(n)
    total = float(np.prod(1.0 + rets) - 1.0)
    mdd = max_drawdown(rets)

    # "give everything a shot": still require some minimal trades to prevent 1-trade champions
    if n < min_trades:
        fitness = -1e9 + n  # still ranks larger n higher
    else:
        # High-score objective. No complexity penalty (per request).
        fitness = (2.0 * sharpe) + (12.0 * total) - (6.0 * mdd)

    return {"n": n, "fitness": float(fitness), "total": total, "sharpe": float(sharpe), "mdd": float(mdd), "mean": mean, "std": std}


def split_by_time(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # train, val, test by time quantiles (stable, easy)
    q1 = df["asof_utc"].quantile(train_frac)
    q2 = df["asof_utc"].quantile(train_frac + val_frac)

    train = df[df["asof_utc"] <= q1]
    val = df[(df["asof_utc"] > q1) & (df["asof_utc"] <= q2)]
    test = df[df["asof_utc"] > q2]
    return train, val, test


def eval_genome(df: pd.DataFrame, g: Genome, train_frac: float, val_frac: float, min_trades: int) -> Dict[str, Any]:
    # scoring uses val for evolution; champion uses test.
    _, val, test = split_by_time(df, train_frac=train_frac, val_frac=val_frac)

    # deterministic randomness per genome (so random strategies don't change every eval)
    seed = int(genome_id(g)) % (2**31 - 1)

    tr_val = trade_returns(val, g, rng_seed=seed)
    val_rets = tr_val["ret"].to_numpy(dtype=float) if not tr_val.empty else np.array([], dtype=float)

    tr_test = trade_returns(test, g, rng_seed=seed + 7)
    test_rets = tr_test["ret"].to_numpy(dtype=float) if not tr_test.empty else np.array([], dtype=float)

    val_score = score_rets(val_rets, min_trades=min_trades)
    test_score = score_rets(test_rets, min_trades=min_trades)

    return {"val": val_score, "test": test_score}


# ============================================================
# Evolution loop
# ============================================================
def evolve_one_cycle(
    df: pd.DataFrame,
    population: List[Genome],
    rng: random.Random,
    cols: List[str],
    cfg: "Config",
) -> Tuple[List[Genome], Dict[str, Any]]:
    pop = population
    best_val: Dict[str, Any] = {}

    for gen in range(cfg.generations_per_cycle):
        # Random immigrants each generation
        immigrants = max(1, int(cfg.pop_size * cfg.random_immigrant_rate))
        for _ in range(immigrants):
            pop[rng.randrange(len(pop))] = random_genome(rng, cols)

        scored: List[Tuple[Dict[str, Any], Genome]] = []
        for g in pop:
            res = eval_genome(df, g, train_frac=cfg.train_frac, val_frac=cfg.val_frac, min_trades=cfg.min_trades)
            scored.append((res, g))

        # Sort by validation fitness for selection/breeding
        scored.sort(key=lambda x: x[0]["val"]["fitness"], reverse=True)
        best_val = {"gen": gen, "res": scored[0][0], "genome": scored[0][1]}

        elite_n = max(2, int(len(pop) * cfg.elite_frac))
        elites = [g for _, g in scored[:elite_n]]

        next_pop: List[Genome] = elites.copy()

        # Breed rest
        while len(next_pop) < len(pop):
            a = rng.choice(elites)
            b = rng.choice(elites)
            child = crossover(a, b, rng)
            child = mutate(child, rng, cols=cols, max_expr_depth=cfg.max_expr_depth)
            next_pop.append(child)

        pop = next_pop

    return pop, best_val


# ============================================================
# Signals + champion logic
# ============================================================
def choose_candidate_for_champion(df: pd.DataFrame, population: List[Genome], cfg: "Config") -> Tuple[Genome, Dict[str, Any]]:
    """
    "Give everything a shot": champion candidate is best TEST fitness
    among the entire population (not just elites).
    """
    best_g = population[0]
    best_res = eval_genome(df, best_g, cfg.train_frac, cfg.val_frac, cfg.min_trades)
    best_score = best_res["test"]["fitness"]

    for g in population[1:]:
        res = eval_genome(df, g, cfg.train_frac, cfg.val_frac, cfg.min_trades)
        s = res["test"]["fitness"]
        if s > best_score:
            best_score = s
            best_g = g
            best_res = res
    return best_g, best_res


def compute_signals(screener_df: pd.DataFrame, g: Genome) -> pd.DataFrame:
    if screener_df.empty:
        return pd.DataFrame()

    d = screener_df.copy()

    # Keep symbol col if exists
    if "symbol" not in d.columns:
        # try common alternative
        for c in d.columns:
            if str(c).strip().lower() == "symbol":
                d.rename(columns={c: "symbol"}, inplace=True)
                break

    # Best-effort numeric conversion for all other columns
    for c in d.columns:
        if c in ("symbol", "asof_utc", "status", "tradeable"):
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Stable RNG for random selection
    seed = int(genome_id(g)) % (2**31 - 1)
    sig = make_signals_for_frame(d, g, rng_seed=seed).to_numpy(dtype=int)
    d["signal"] = sig
    d = d[d["signal"] != 0].copy()
    if d.empty:
        return d

    d["side"] = np.where(d["signal"] == 1, "LONG", "SHORT")
    d["score_expr"] = expr_to_str(g.score_expr)
    d["select_mode"] = g.select_mode
    d["thr"] = g.thr
    d["top_k"] = g.top_k
    d["horizon"] = g.horizon

    cols = [c for c in ["symbol", "side", "signal", "close", "spread_bps", "asof_utc", "select_mode", "thr", "top_k", "horizon", "score_expr"] if c in d.columns]
    return d[cols].sort_values(["symbol"])


# ============================================================
# State persistence
# ============================================================
def load_state(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path: str, state: Dict[str, Any]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


# ============================================================
# Config + main
# ============================================================
@dataclass
class Config:
    data_source: str  # google|excel
    excel_path: str

    sheet_id: str
    screener_tab: str
    history_tab: str
    strategies_tab: str
    signals_tab: str
    champion_tab: str

    sleep_seconds: int
    history_lookback_rows: int

    pop_size: int
    generations_per_cycle: int
    elite_frac: float
    random_immigrant_rate: float
    max_expr_depth: int

    train_frac: float
    val_frac: float
    min_trades: int

    seed: int
    state_path: str

    signals_source: str  # champion|best_val


def load_config() -> Config:
    return Config(
        data_source=(env_str("DATA_SOURCE", "google") or "google").lower(),
        excel_path=env_str("EXCEL_PATH", "./Oanda.xlsx") or "./Oanda.xlsx",
        sheet_id=env_str("SHEET_ID", "") or "",
        screener_tab=env_str("SCREENER_TAB", "Screener") or "Screener",
        history_tab=env_str("HISTORY_TAB", "history") or "history",
        strategies_tab=env_str("STRATEGIES_TAB", "strategies") or "strategies",
        signals_tab=env_str("SIGNALS_TAB", "signals") or "signals",
        champion_tab=env_str("CHAMPION_TAB", "champion") or "champion",
        sleep_seconds=env_int("SLEEP_SECONDS", 60),
        history_lookback_rows=env_int("HISTORY_LOOKBACK_ROWS", 80000),
        pop_size=env_int("POP_SIZE", 96),
        generations_per_cycle=env_int("GENERATIONS_PER_CYCLE", 6),
        elite_frac=env_float("ELITE_FRAC", 0.22),
        random_immigrant_rate=env_float("RANDOM_IMMIGRANT_RATE", 0.07),
        max_expr_depth=env_int("MAX_EXPR_DEPTH", 6),
        train_frac=env_float("TRAIN_FRAC", 0.70),
        val_frac=env_float("VAL_FRAC", 0.15),
        min_trades=env_int("MIN_TRADES", 25),
        seed=env_int("SEED", 1337),
        state_path=env_str("STATE_PATH", "/data/state.json") or "/data/state.json",
        signals_source=(env_str("SIGNALS_SOURCE", "champion") or "champion").lower(),
    )


def make_client(cfg: Config) -> SheetClient:
    if cfg.data_source == "excel":
        return ExcelClient(cfg.excel_path, cfg.screener_tab, cfg.history_tab)

    if not cfg.sheet_id:
        raise RuntimeError("DATA_SOURCE=google requires SHEET_ID.")
    return GoogleSheetsClient(
        sheet_id=cfg.sheet_id,
        screener_tab=cfg.screener_tab,
        history_tab=cfg.history_tab,
        strategies_tab=cfg.strategies_tab,
        signals_tab=cfg.signals_tab,
        champion_tab=cfg.champion_tab,
    )


def extract_feature_cols(df: pd.DataFrame) -> List[str]:
    # Choose columns that are plausibly numeric features (exclude keys and forward returns)
    cols: List[str] = []
    for c in df.columns:
        if c in ("symbol", "asof_utc", "status", "tradeable"):
            continue
        if str(c).startswith("fwd_ret_"):
            continue
        # prefer columns that have at least some numeric values
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, int(0.01 * len(df))):
            cols.append(c)
    return cols


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit.")
    args = parser.parse_args()

    cfg = load_config()
    rng = random.Random(cfg.seed)
    client = make_client(cfg)
    state = load_state(cfg.state_path)

    # Load history
    hist_raw = client.read_history(cfg.history_lookback_rows)
    hist = coerce_types(hist_raw)
    if hist.empty:
        log.warning("No history rows yet. Exiting." if args.once else "No history rows yet; will retry.")
        if args.once:
            return
        while hist.empty:
            time.sleep(cfg.sleep_seconds)
            hist_raw = client.read_history(cfg.history_lookback_rows)
            hist = coerce_types(hist_raw)

    # Build feature list for random programs
    feature_cols = extract_feature_cols(hist)
    if not feature_cols:
        raise RuntimeError("No usable numeric feature columns found in history.")

    # Load or init population
    pop: List[Genome] = []
    if "population" in state:
        try:
            pop = [Genome(**p).normalize() for p in state["population"]]
        except Exception:
            pop = []
    if not pop or len(pop) != cfg.pop_size:
        pop = [random_genome(rng, feature_cols) for _ in range(cfg.pop_size)]

    # Champion
    champ: Optional[Genome] = None
    champ_score = -1e18
    if "champion" in state:
        try:
            champ = Genome(**state["champion"]["genome"]).normalize()
            champ_score = float(state["champion"]["score"])
        except Exception:
            champ = None
            champ_score = -1e18

    # Strategy log header
    strategies_header = [
        "ts_utc",
        "cycle",
        "best_val_strategy_id",
        "best_val_fitness",
        "best_val_trades",
        "best_val_total",
        "best_val_sharpe",
        "best_val_mdd",
        "best_val_select_mode",
        "best_val_expr",
        "best_test_strategy_id",
        "best_test_fitness",
        "best_test_trades",
        "best_test_total",
        "best_test_sharpe",
        "best_test_mdd",
        "champion_strategy_id",
        "champion_test_fitness",
        "champion_expr",
        "params_json",
    ]

    champion_header = [
        "ts_utc",
        "champion_strategy_id",
        "champion_test_fitness",
        "champion_test_trades",
        "champion_test_total",
        "champion_test_sharpe",
        "champion_test_mdd",
        "select_mode",
        "thr",
        "top_k",
        "horizon",
        "filters_count",
        "score_expr",
        "params_json",
    ]

    cycle = int(state.get("cycle", 0))

    log.info("Starting evolver v2: pop_size=%s features=%s train/val/test=%s/%s/%s",
             cfg.pop_size, len(feature_cols), cfg.train_frac, cfg.val_frac, 1.0 - cfg.train_frac - cfg.val_frac)

    while True:
        # refresh history each loop
        hist_raw = client.read_history(cfg.history_lookback_rows)
        hist = coerce_types(hist_raw)
        if hist.empty:
            log.warning("History empty; sleeping %ss", cfg.sleep_seconds)
            if args.once:
                return
            time.sleep(cfg.sleep_seconds)
            continue

        # update feature cols periodically (new columns, etc.)
        feature_cols = extract_feature_cols(hist) or feature_cols

        # horizons used: ensure include current population horizons
        horizons = sorted(set([g.horizon for g in pop] + ([champ.horizon] if champ else []) + [1, 2, 3, 6, 12]))
        hist2 = add_forward_returns(hist, horizons)

        # evolve
        pop, best_val = evolve_one_cycle(hist2, pop, rng, feature_cols, cfg)

        best_val_g: Genome = best_val["genome"]
        best_val_res = best_val["res"]

        # champion candidate: best TEST score among entire population
        best_test_g, best_test_res = choose_candidate_for_champion(hist2, pop, cfg)

        # update champion if beaten
        improved = False
        if best_test_res["test"]["fitness"] > champ_score:
            champ = best_test_g
            champ_score = float(best_test_res["test"]["fitness"])
            improved = True

        champ_id = genome_id(champ) if champ else ""
        champ_expr = expr_to_str(champ.score_expr) if champ else ""
        champ_test = eval_genome(hist2, champ, cfg.train_frac, cfg.val_frac, cfg.min_trades)["test"] if champ else {"fitness": -1e18, "n": 0, "total": 0.0, "sharpe": 0.0, "mdd": 0.0}

        cycle += 1
        state["cycle"] = cycle
        state["population"] = [asdict(g) for g in pop]
        if champ:
            state["champion"] = {"genome": asdict(champ), "score": champ_score}
        save_state(cfg.state_path, state)

        # log strategies row
        row = [
            now_utc_iso(),
            cycle,
            genome_id(best_val_g),
            float(best_val_res["val"]["fitness"]),
            int(best_val_res["val"]["n"]),
            float(best_val_res["val"]["total"]),
            float(best_val_res["val"]["sharpe"]),
            float(best_val_res["val"]["mdd"]),
            best_val_g.select_mode,
            expr_to_str(best_val_g.score_expr),
            genome_id(best_test_g),
            float(best_test_res["test"]["fitness"]),
            int(best_test_res["test"]["n"]),
            float(best_test_res["test"]["total"]),
            float(best_test_res["test"]["sharpe"]),
            float(best_test_res["test"]["mdd"]),
            champ_id,
            float(champ_test["fitness"]),
            champ_expr,
            json.dumps(
                {
                    "best_val": asdict(best_val_g),
                    "best_test": asdict(best_test_g),
                    "champion": asdict(champ) if champ else None,
                },
                sort_keys=True,
            ),
        ]
        client.append_strategies(strategies_header, [row])

        if improved and champ:
            champ_row = [
                now_utc_iso(),
                champ_id,
                float(champ_test["fitness"]),
                int(champ_test["n"]),
                float(champ_test["total"]),
                float(champ_test["sharpe"]),
                float(champ_test["mdd"]),
                champ.select_mode,
                champ.thr,
                champ.top_k,
                champ.horizon,
                len(champ.filters),
                champ_expr,
                json.dumps(asdict(champ), sort_keys=True),
            ]
            client.write_champion(champion_header, champ_row)

        # write signals from champion or best_val
        signal_g = champ if (cfg.signals_source == "champion" and champ) else best_val_g
        screener = client.read_screener()
        signals = compute_signals(screener, signal_g)

        header = ["ts_utc", "strategy_id", "source"] + list(signals.columns)
        rows: List[List[Any]] = []
        sid = genome_id(signal_g)
        src = "champion" if (signal_g is champ) else "best_val"
        for _, r in signals.iterrows():
            rows.append([now_utc_iso(), sid, src] + [r.get(c, "") for c in signals.columns])
        client.replace_signals(header, rows)

        log.info(
            "cycle=%s best_val=%s val_fit=%.3f best_test=%s test_fit=%.3f champion=%s champ_fit=%.3f improved=%s",
            cycle,
            genome_id(best_val_g),
            float(best_val_res["val"]["fitness"]),
            genome_id(best_test_g),
            float(best_test_res["test"]["fitness"]),
            champ_id,
            float(champ_test["fitness"]),
            improved,
        )

        if args.once:
            return

        time.sleep(cfg.sleep_seconds)


if __name__ == "__main__":
    main()