import os
import json
import time
import math
import base64
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
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


# -----------------------------
# Strategy genome (simple, evolvable parameter set)
# -----------------------------
@dataclass
class Genome:
    # Mode:
    # - "trend": trade in the direction of distance-from-SMA50
    # - "mr": mean reversion (opposite direction)
    mode: str  # "trend" | "mr"

    # Thresholds (dist_sma50_pct appears to be in "percent points": 1.0 = 1%)
    dist_thr: float
    vol_z_min: float

    # Cost/quality filters
    spread_bps_max: float
    spread_vs_atr_max: float
    atr_mult_range_min: float

    # Optional trend filter using trend_200 sign
    trend_req: bool

    # Forward-return horizon in number of history snapshots
    horizon: int

    def normalize(self) -> "Genome":
        self.mode = "trend" if self.mode not in ("trend", "mr") else self.mode
        self.dist_thr = clamp(float(self.dist_thr), 0.05, 5.0)
        self.vol_z_min = clamp(float(self.vol_z_min), -5.0, 10.0)
        self.spread_bps_max = clamp(float(self.spread_bps_max), 0.1, 200.0)
        self.spread_vs_atr_max = clamp(float(self.spread_vs_atr_max), 0.0001, 5.0)
        self.atr_mult_range_min = clamp(float(self.atr_mult_range_min), 0.0, 50.0)
        self.trend_req = bool(self.trend_req)
        self.horizon = int(clamp(int(self.horizon), 1, 200))
        return self


def random_genome(rng: random.Random) -> Genome:
    g = Genome(
        mode=rng.choice(["trend", "mr"]),
        dist_thr=rng.uniform(0.2, 1.5),
        vol_z_min=rng.uniform(0.2, 2.0),
        spread_bps_max=rng.uniform(2.0, 15.0),
        spread_vs_atr_max=rng.uniform(0.03, 0.15),
        atr_mult_range_min=rng.uniform(0.6, 1.3),
        trend_req=rng.random() < 0.6,
        horizon=rng.choice([1, 2, 3, 6]),
    )
    return g.normalize()


def mutate(g: Genome, rng: random.Random, rate: float = 0.25) -> Genome:
    h = Genome(**asdict(g))
    if rng.random() < rate:
        h.mode = "mr" if h.mode == "trend" else "trend"
    if rng.random() < rate:
        h.dist_thr += rng.gauss(0, 0.20)
    if rng.random() < rate:
        h.vol_z_min += rng.gauss(0, 0.25)
    if rng.random() < rate:
        h.spread_bps_max += rng.gauss(0, 1.25)
    if rng.random() < rate:
        h.spread_vs_atr_max += rng.gauss(0, 0.015)
    if rng.random() < rate:
        h.atr_mult_range_min += rng.gauss(0, 0.07)
    if rng.random() < rate:
        h.trend_req = not h.trend_req
    if rng.random() < rate:
        h.horizon = rng.choice([1, 2, 3, 6, 12])
    return h.normalize()


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    da = asdict(a)
    db = asdict(b)
    child = {}
    for k in da.keys():
        child[k] = da[k] if rng.random() < 0.5 else db[k]
    return Genome(**child).normalize()


# -----------------------------
# Data sources (Google Sheets OR Excel)
# -----------------------------
class SheetClient:
    def read_screener(self) -> pd.DataFrame:
        raise NotImplementedError

    def read_history(self, lookback_rows: int) -> pd.DataFrame:
        raise NotImplementedError

    def append_strategies(self, rows: List[List[Any]]) -> None:
        raise NotImplementedError

    def replace_signals(self, header: List[str], rows: List[List[Any]]) -> None:
        raise NotImplementedError


class ExcelClient(SheetClient):
    def __init__(self, path: str, screener_tab: str, history_tab: str):
        self.path = path
        self.screener_tab = screener_tab
        self.history_tab = history_tab

    def read_screener(self) -> pd.DataFrame:
        return pd.read_excel(self.path, sheet_name=self.screener_tab)

    def read_history(self, lookback_rows: int) -> pd.DataFrame:
        # history tab is append-only with no header row (per your file)
        hist_raw = pd.read_excel(self.path, sheet_name=self.history_tab, header=None)
        scr = self.read_screener()
        hist_raw = hist_raw.tail(lookback_rows) if lookback_rows > 0 else hist_raw
        hist_raw.columns = list(scr.columns)
        return hist_raw

    def append_strategies(self, rows: List[List[Any]]) -> None:
        # Excel doesn't support appending in-place on Railway; print for local testing
        for r in rows:
            print("STRATEGY_ROW", r)

    def replace_signals(self, header: List[str], rows: List[List[Any]]) -> None:
        print("SIGNALS_HEADER", header)
        for r in rows:
            print("SIGNAL_ROW", r)


class GoogleSheetsClient(SheetClient):
    def __init__(
        self,
        sheet_id: str,
        screener_tab: str,
        history_tab: str,
        strategies_tab: str,
        signals_tab: str,
    ):
        import gspread
        from google.oauth2 import service_account

        sa_json = env_str("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not sa_json:
            raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON env var.")

        # Allow either raw JSON or base64-encoded JSON
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

    def _ws(self, name: str, rows: int = 1000, cols: int = 40):
        import gspread

        try:
            return self.sh.worksheet(name)
        except gspread.WorksheetNotFound:
            return self.sh.add_worksheet(title=name, rows=str(rows), cols=str(cols))

    def read_screener(self) -> pd.DataFrame:
        ws = self._ws(self.screener_tab, rows=2000, cols=120)
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame()
        header = values[0]
        rows = values[1:]
        return pd.DataFrame(rows, columns=header)

    def read_history(self, lookback_rows: int) -> pd.DataFrame:
        ws = self._ws(self.history_tab, rows=50000, cols=120)
        values = ws.get_all_values()
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

    def append_strategies(self, rows: List[List[Any]]) -> None:
        ws = self._ws(self.strategies_tab, rows=5000, cols=30)
        if not ws.get_all_values():
            ws.append_row(
                [
                    "ts_utc",
                    "generation",
                    "strategy_id",
                    "fitness",
                    "val_trades",
                    "val_total",
                    "val_sharpe",
                    "val_mdd",
                    "params_json",
                ],
                value_input_option="RAW",
            )
        ws.append_rows(rows, value_input_option="RAW")

    def replace_signals(self, header: List[str], rows: List[List[Any]]) -> None:
        ws = self._ws(self.signals_tab, rows=2000, cols=max(20, len(header) + 2))
        ws.clear()
        ws.append_row(header, value_input_option="RAW")
        if rows:
            ws.append_rows(rows, value_input_option="RAW")


# -----------------------------
# Backtest + evolve
# -----------------------------
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Drop blank separator rows (your sample history uses them)
    if "symbol" in d.columns:
        d = d[d["symbol"].notna()].copy()

    # Datetime
    if "asof_utc" in d.columns:
        d["asof_utc"] = pd.to_datetime(d["asof_utc"], errors="coerce", utc=True)
        d = d[d["asof_utc"].notna()].copy()

    # Numerics (subset we use)
    for c in ["close", "spread_bps", "spread_vs_atr", "atr_mult_range", "vol_z_20d", "dist_sma50_pct", "trend_200"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    return d


def add_forward_returns(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    d = df.sort_values(["symbol", "asof_utc"]).reset_index(drop=True)
    for h in horizons:
        d[f"fwd_ret_{h}"] = d.groupby("symbol")["close"].shift(-h) / d["close"] - 1.0
    return d


def strategy_trades(df: pd.DataFrame, g: Genome) -> pd.DataFrame:
    h = g.horizon
    col = f"fwd_ret_{h}"
    if col not in df.columns:
        return df.iloc[0:0].copy()

    d = df.copy()

    # Optional tradeable/status filters
    if "tradeable" in d.columns:
        d = d[d["tradeable"].map(to_bool)]
    if "status" in d.columns:
        d = d[d["status"].astype(str).str.lower().eq("tradeable")]

    # Filters + required fields
    for need in ["spread_bps", "spread_vs_atr", "atr_mult_range", "vol_z_20d", "dist_sma50_pct", col]:
        d = d[d[need].notna()]

    d = d[d["spread_bps"] <= g.spread_bps_max]
    d = d[d["spread_vs_atr"] <= g.spread_vs_atr_max]
    d = d[d["atr_mult_range"] >= g.atr_mult_range_min]
    d = d[d["vol_z_20d"] >= g.vol_z_min]

    dist = d["dist_sma50_pct"]
    trend = d["trend_200"] if "trend_200" in d.columns else pd.Series(0.0, index=d.index)

    long_cond = dist >= g.dist_thr
    short_cond = dist <= -g.dist_thr

    if g.mode == "trend":
        sig = np.where(long_cond, 1, np.where(short_cond, -1, 0))
        if g.trend_req and "trend_200" in d.columns:
            sig = np.where((sig == 1) & (trend > 0), 1, np.where((sig == -1) & (trend < 0), -1, 0))
    else:
        sig = np.where(long_cond, -1, np.where(short_cond, 1, 0))

    d = d.assign(signal=sig)
    d = d[d["signal"] != 0]

    # Cost model: ~1 spread per round trip => spread_bps in return units
    cost = d["spread_bps"] / 10000.0
    ret = d["signal"].astype(float) * d[col].astype(float) - cost

    return d.assign(ret=ret)[["asof_utc", "symbol", "signal", col, "spread_bps", "ret"]]


def eval_genome(df: pd.DataFrame, g: Genome, train_frac: float, min_val_trades: int) -> Dict[str, Any]:
    if df.empty:
        return {"fitness": -1e9, "val": {"n": 0}}

    cutoff = df["asof_utc"].quantile(train_frac)
    val = df[df["asof_utc"] > cutoff]

    tr_val = strategy_trades(val, g)
    rets = tr_val["ret"].to_numpy(dtype=float) if not tr_val.empty else np.array([], dtype=float)
    n = len(rets)
    if n == 0:
        return {"fitness": -1e9, "val": {"n": 0}}

    mean = float(np.mean(rets))
    std = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    sharpe = 0.0 if std == 0.0 else mean / std * math.sqrt(n)
    total = float(np.prod(1.0 + rets) - 1.0)
    mdd = max_drawdown(rets)

    if n < min_val_trades:
        fitness = -1e6 + n
    else:
        fitness = (1.5 * sharpe) + (5.0 * total) - (3.0 * mdd) - (g.spread_bps_max / 1000.0)

    return {"fitness": fitness, "val": {"n": n, "mean": mean, "std": std, "sharpe": sharpe, "total": total, "mdd": mdd}}


def evolve(
    df: pd.DataFrame,
    population: List[Genome],
    rng: random.Random,
    generations: int,
    elite_frac: float,
    train_frac: float,
    min_val_trades: int,
) -> Tuple[List[Genome], Dict[str, Any]]:
    pop = population
    best_summary: Dict[str, Any] = {}

    for gen in range(generations):
        scored = [(eval_genome(df, g, train_frac, min_val_trades), g) for g in pop]
        scored.sort(key=lambda x: x[0]["fitness"], reverse=True)

        best = scored[0]
        best_summary = {"gen": gen, "fitness": best[0]["fitness"], "val": best[0]["val"], "genome": best[1]}

        elite_n = max(2, int(len(pop) * elite_frac))
        elites = [g for _, g in scored[:elite_n]]

        next_pop: List[Genome] = elites.copy()
        while len(next_pop) < len(pop):
            a = rng.choice(elites)
            b = rng.choice(elites)
            child = mutate(crossover(a, b, rng), rng)
            next_pop.append(child)

        pop = next_pop

    return pop, best_summary


def genome_id(g: Genome) -> str:
    payload = json.dumps(asdict(g), sort_keys=True)
    return str(abs(hash(payload)))[:10]


def compute_signals(screener_df: pd.DataFrame, g: Genome) -> pd.DataFrame:
    if screener_df.empty:
        return pd.DataFrame()

    d = screener_df.copy()
    for c in ["dist_sma50_pct", "vol_z_20d", "spread_bps", "spread_vs_atr", "atr_mult_range", "trend_200", "close"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    for need in ["spread_bps", "spread_vs_atr", "atr_mult_range", "vol_z_20d", "dist_sma50_pct"]:
        d = d[d[need].notna()]

    d = d[d["spread_bps"] <= g.spread_bps_max]
    d = d[d["spread_vs_atr"] <= g.spread_vs_atr_max]
    d = d[d["atr_mult_range"] >= g.atr_mult_range_min]
    d = d[d["vol_z_20d"] >= g.vol_z_min]

    dist = d["dist_sma50_pct"]
    trend = d["trend_200"] if "trend_200" in d.columns else pd.Series(0.0, index=d.index)

    long_cond = dist >= g.dist_thr
    short_cond = dist <= -g.dist_thr

    if g.mode == "trend":
        sig = np.where(long_cond, 1, np.where(short_cond, -1, 0))
        if g.trend_req and "trend_200" in d.columns:
            sig = np.where((sig == 1) & (trend > 0), 1, np.where((sig == -1) & (trend < 0), -1, 0))
    else:
        sig = np.where(long_cond, -1, np.where(short_cond, 1, 0))

    d = d.assign(signal=sig)
    d = d[d["signal"] != 0].copy()

    d["side"] = np.where(d["signal"] == 1, "LONG", "SHORT")
    d["reason"] = f"mode={g.mode}, dist_thr={g.dist_thr:.3f}, vol_z_min={g.vol_z_min:.3f}"

    cols = [c for c in ["symbol", "side", "close", "dist_sma50_pct", "vol_z_20d", "spread_bps", "asof_utc", "reason"] if c in d.columns]
    return d[cols].sort_values(["spread_bps", "symbol"])


# -----------------------------
# State persistence
# -----------------------------
def load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


# -----------------------------
# Main loop
# -----------------------------
@dataclass
class Config:
    data_source: str  # "google" | "excel"
    excel_path: str

    sheet_id: str
    screener_tab: str
    history_tab: str
    strategies_tab: str
    signals_tab: str

    sleep_seconds: int
    history_lookback_rows: int

    pop_size: int
    generations_per_cycle: int
    elite_frac: float

    train_frac: float
    min_val_trades: int

    seed: int
    state_path: str


def load_config() -> Config:
    return Config(
        data_source=env_str("DATA_SOURCE", "google").lower(),
        excel_path=env_str("EXCEL_PATH", "./Oanda.xlsx") or "./Oanda.xlsx",
        sheet_id=env_str("SHEET_ID", "") or "",
        screener_tab=env_str("SCREENER_TAB", "Screener") or "Screener",
        history_tab=env_str("HISTORY_TAB", "history") or "history",
        strategies_tab=env_str("STRATEGIES_TAB", "strategies") or "strategies",
        signals_tab=env_str("SIGNALS_TAB", "signals") or "signals",
        sleep_seconds=env_int("SLEEP_SECONDS", 60),
        history_lookback_rows=env_int("HISTORY_LOOKBACK_ROWS", 50000),
        pop_size=env_int("POP_SIZE", 64),
        generations_per_cycle=env_int("GENERATIONS_PER_CYCLE", 5),
        elite_frac=env_float("ELITE_FRAC", 0.25),
        train_frac=env_float("TRAIN_FRAC", 0.8),
        min_val_trades=env_int("MIN_VAL_TRADES", 25),
        seed=env_int("SEED", 1337),
        state_path=env_str("STATE_PATH", "./state/state.json") or "./state/state.json",
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run a single evolve+signal cycle and exit.")
    args = parser.parse_args()

    cfg = load_config()
    rng = random.Random(cfg.seed)

    client = make_client(cfg)
    state = load_state(cfg.state_path)

    # Load or initialize population
    pop: List[Genome] = []
    if "population" in state:
        try:
            pop = [Genome(**p).normalize() for p in state["population"]]
        except Exception:
            pop = []
    if not pop or len(pop) != cfg.pop_size:
        pop = [random_genome(rng) for _ in range(cfg.pop_size)]

    horizons = sorted(set([g.horizon for g in pop] + [1, 2, 3, 6]))

    while True:
        hist_raw = client.read_history(cfg.history_lookback_rows)
        hist = coerce_types(hist_raw)

        if hist.empty:
            print("No history rows yet. Sleeping…")
            time.sleep(cfg.sleep_seconds)
            if args.once:
                return
            continue

        hist = add_forward_returns(hist, horizons)

        pop, best = evolve(
            hist,
            pop,
            rng=rng,
            generations=cfg.generations_per_cycle,
            elite_frac=cfg.elite_frac,
            train_frac=cfg.train_frac,
            min_val_trades=cfg.min_val_trades,
        )
        best_g: Genome = best["genome"]

        sid = genome_id(best_g)
        row = [
            now_utc_iso(),
            int(state.get("generation", 0)) + 1,
            sid,
            float(best["fitness"]),
            int(best["val"]["n"]),
            float(best["val"]["total"]),
            float(best["val"]["sharpe"]),
            float(best["val"]["mdd"]),
            json.dumps(asdict(best_g), sort_keys=True),
        ]
        client.append_strategies([row])

        screener = client.read_screener()
        signals = compute_signals(screener, best_g)

        header = ["ts_utc", "strategy_id"] + list(signals.columns)
        rows: List[List[Any]] = []
        for _, r in signals.iterrows():
            rows.append([now_utc_iso(), sid] + [r.get(c, "") for c in signals.columns])
        client.replace_signals(header, rows)

        state["generation"] = int(state.get("generation", 0)) + 1
        state["population"] = [asdict(g) for g in pop]
        state["best"] = {"strategy_id": sid, "params": asdict(best_g), "fitness": best["fitness"], "val": best["val"]}
        save_state(cfg.state_path, state)

        print(f"[cycle] gen={state['generation']} best_id={sid} fitness={best['fitness']:.4f} val_trades={best['val']['n']}")

        if args.once:
            return

        time.sleep(cfg.sleep_seconds)


if __name__ == "__main__":
    main()
