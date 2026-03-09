
import os
import json
import time
import math
import base64
import random
import argparse
import logging
import hashlib
from dataclasses import dataclass, asdict, field
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


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t", "on")


def env_csv(name: str, default: str) -> List[str]:
    raw = env_str(name, default) or default
    return [x.strip() for x in raw.split(",") if x.strip()]


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
VALID_TIERS = ("simple", "medium", "complex")
VALID_FAMILIES = ("expr", "template")
VALID_TEMPLATE_KINDS = ("feature_rank", "feature_spread", "feature_threshold")


def _nan_array(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def _safe_pow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.power(a, b)
    out[~np.isfinite(out)] = np.nan
    return out


def allowed_unary_ops_for_tier(tier: str) -> Tuple[str, ...]:
    if tier == "simple":
        return ("abs", "neg", "tanh")
    if tier == "medium":
        return ("abs", "neg", "tanh", "log1p", "sqrt", "clip")
    return UNARY_OPS


def allowed_binary_ops_for_tier(tier: str) -> Tuple[str, ...]:
    if tier == "simple":
        return ("add", "sub", "mul", "div", "max", "min")
    if tier == "medium":
        return ("add", "sub", "mul", "div", "max", "min")
    return BINARY_OPS


def max_num_depth_for_tier(tier: str, max_expr_depth: int) -> int:
    if tier == "simple":
        return min(2, max_expr_depth)
    if tier == "medium":
        return min(3, max_expr_depth)
    return max(2, max_expr_depth)


def max_bool_depth_for_tier(tier: str, max_expr_depth: int) -> int:
    if tier == "simple":
        return 1
    if tier == "medium":
        return min(2, max_expr_depth)
    return min(3, max_expr_depth)


def max_filters_for_tier(tier: str) -> int:
    if tier == "simple":
        return 1
    if tier == "medium":
        return 2
    return 5


def family_weights_for_tier(tier: str) -> Tuple[float, float]:
    # template, expr
    if tier == "simple":
        return (0.75, 0.25)
    if tier == "medium":
        return (0.45, 0.55)
    return (0.15, 0.85)


def select_mode_weights_for_tier(tier: str) -> List[float]:
    if tier == "simple":
        return [0.45, 0.50, 0.05]  # threshold, topk, random
    if tier == "medium":
        return [0.55, 0.35, 0.10]
    return [0.55, 0.30, 0.15]


def random_const(rng: random.Random) -> float:
    if rng.random() < 0.85:
        return rng.gauss(0, 1.0)
    return rng.gauss(0, 5.0)


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
        cond = eval_bool(
            node.get(
                "cond",
                {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}},
            ),
            df,
        )
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
        mask = ~np.isnan(a) & ~np.isnan(b)
        if op == "gt":
            return np.greater(a, b, where=mask, out=np.zeros(n, dtype=bool))
        if op == "lt":
            return np.less(a, b, where=mask, out=np.zeros(n, dtype=bool))
        if op == "ge":
            return np.greater_equal(a, b, where=mask, out=np.zeros(n, dtype=bool))
        if op == "le":
            return np.less_equal(a, b, where=mask, out=np.zeros(n, dtype=bool))
        return np.zeros(n, dtype=bool)

    if t == "log":
        op = node.get("op")
        a = eval_bool(node.get("a", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), df)
        b = eval_bool(node.get("b", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), df)
        if op == "and":
            return a & b
        if op == "or":
            return a | b
        return a

    if t == "not":
        a = eval_bool(node.get("a", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), df)
        return ~a

    arr = eval_num(node, df)
    return np.isfinite(arr) & (arr > 0)


def expr_to_str(node: Dict[str, Any]) -> str:
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


def expr_node_count(node: Dict[str, Any]) -> int:
    if not isinstance(node, dict):
        return 0
    t = node.get("t")
    if t in ("col", "const"):
        return 1
    if t in ("u", "not"):
        return 1 + expr_node_count(node.get("a", {}))
    if t in ("b", "cmp", "log"):
        return 1 + expr_node_count(node.get("a", {})) + expr_node_count(node.get("b", {}))
    if t == "if":
        return 1 + expr_node_count(node.get("cond", {})) + expr_node_count(node.get("x", {})) + expr_node_count(node.get("y", {}))
    return 1


def random_num_expr(rng: random.Random, cols: List[str], depth: int, tier: str) -> Dict[str, Any]:
    unary_ops = allowed_unary_ops_for_tier(tier)
    binary_ops = allowed_binary_ops_for_tier(tier)

    if depth <= 0 or rng.random() < (0.55 if tier == "simple" else 0.35):
        if cols and rng.random() < 0.75:
            return {"t": "col", "name": rng.choice(cols)}
        return {"t": "const", "v": random_const(rng)}

    if tier != "simple" and rng.random() < (0.08 if tier == "medium" else 0.12) and depth >= 1:
        return {
            "t": "if",
            "cond": random_bool_expr(rng, cols, depth - 1, tier),
            "x": random_num_expr(rng, cols, depth - 1, tier),
            "y": random_num_expr(rng, cols, depth - 1, tier),
        }

    if rng.random() < 0.45:
        op = rng.choice(unary_ops)
        node = {"t": "u", "op": op, "a": random_num_expr(rng, cols, depth - 1, tier)}
        if op == "clip":
            node["lo"] = random_const(rng)
            node["hi"] = node["lo"] + abs(random_const(rng)) + 0.5
        return node

    op = rng.choice(binary_ops)
    return {
        "t": "b",
        "op": op,
        "a": random_num_expr(rng, cols, depth - 1, tier),
        "b": random_num_expr(rng, cols, depth - 1, tier),
    }


def random_bool_expr(rng: random.Random, cols: List[str], depth: int, tier: str) -> Dict[str, Any]:
    if depth <= 0 or rng.random() < (0.75 if tier == "simple" else 0.55):
        return {
            "t": "cmp",
            "op": rng.choice(CMP_OPS),
            "a": random_num_expr(rng, cols, 0, tier),
            "b": random_num_expr(rng, cols, 0, tier),
        }
    if tier != "simple" and rng.random() < 0.15:
        return {"t": "not", "a": random_bool_expr(rng, cols, depth - 1, tier)}
    return {
        "t": "log",
        "op": rng.choice(LOG_OPS),
        "a": random_bool_expr(rng, cols, depth - 1, tier),
        "b": random_bool_expr(rng, cols, depth - 1, tier),
    }


def simplify_num_expr(node: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    t = node.get("t")
    if t in ("col", "const"):
        return node
    if t == "u":
        return node.get("a", {"t": "const", "v": 0.0})
    if t == "b":
        return node.get("a", {"t": "const", "v": 0.0}) if rng.random() < 0.5 else node.get("b", {"t": "const", "v": 0.0})
    if t == "if":
        return node.get("x", {"t": "const", "v": 0.0}) if rng.random() < 0.5 else node.get("y", {"t": "const", "v": 0.0})
    return {"t": "const", "v": 0.0}


def simplify_bool_expr(node: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    t = node.get("t")
    if t == "cmp":
        return node
    if t == "log":
        return node.get("a", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}) if rng.random() < 0.5 else node.get("b", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}})
    if t == "not":
        return node.get("a", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}})
    return {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}


def mutate_expr_any(node: Dict[str, Any], rng: random.Random, cols: List[str], tier: str, max_depth: int) -> Dict[str, Any]:
    unary_ops = allowed_unary_ops_for_tier(tier)
    binary_ops = allowed_binary_ops_for_tier(tier)
    simplify_prob = 0.20 if tier == "simple" else (0.12 if tier == "medium" else 0.08)
    wrap_prob = 0.05 if tier == "simple" else (0.10 if tier == "medium" else 0.15)
    compose_prob = 0.05 if tier == "simple" else (0.08 if tier == "medium" else 0.12)

    if rng.random() < 0.08:
        return random_num_expr(rng, cols, max_depth, tier)

    t = node.get("t")

    if rng.random() < simplify_prob:
        return simplify_num_expr(node, rng)

    if t in ("col", "const") and rng.random() < 0.30:
        return random_num_expr(rng, cols, 1 if tier == "simple" else 2, tier)

    out = dict(node)

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
        if rng.random() < 0.25:
            out["op"] = rng.choice(unary_ops)
        out["a"] = mutate_expr_any(out.get("a", {"t": "const", "v": 0.0}), rng, cols, tier, max(0, max_depth - 1))
        if out.get("op") == "clip":
            if rng.random() < 0.5:
                out["lo"] = random_const(rng)
            if rng.random() < 0.5:
                out["hi"] = float(out.get("lo", -1.0)) + abs(random_const(rng)) + 0.5
        if rng.random() < wrap_prob:
            out = {"t": "u", "op": rng.choice(unary_ops), "a": out}
        return out

    if t == "b":
        if rng.random() < 0.25:
            out["op"] = rng.choice(binary_ops)
        out["a"] = mutate_expr_any(out.get("a", {"t": "const", "v": 0.0}), rng, cols, tier, max(0, max_depth - 1))
        out["b"] = mutate_expr_any(out.get("b", {"t": "const", "v": 1.0}), rng, cols, tier, max(0, max_depth - 1))
        if rng.random() < compose_prob:
            out = {
                "t": "b",
                "op": rng.choice(binary_ops),
                "a": out,
                "b": random_num_expr(rng, cols, 1 if tier == "simple" else 2, tier),
            }
        return out

    if t == "if":
        out["cond"] = mutate_bool_any(
            out.get("cond", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}),
            rng,
            cols,
            tier,
            max(0, max_depth - 1),
        )
        out["x"] = mutate_expr_any(out.get("x", {"t": "const", "v": 1.0}), rng, cols, tier, max(0, max_depth - 1))
        out["y"] = mutate_expr_any(out.get("y", {"t": "const", "v": -1.0}), rng, cols, tier, max(0, max_depth - 1))
        return out

    return random_num_expr(rng, cols, max_depth, tier)


def mutate_bool_any(node: Dict[str, Any], rng: random.Random, cols: List[str], tier: str, max_depth: int) -> Dict[str, Any]:
    if rng.random() < 0.08:
        return random_bool_expr(rng, cols, max_depth, tier)

    if rng.random() < (0.20 if tier == "simple" else 0.10):
        return simplify_bool_expr(node, rng)

    t = node.get("t")
    out = dict(node)

    if t == "cmp":
        if rng.random() < 0.30:
            out["op"] = rng.choice(CMP_OPS)
        out["a"] = mutate_expr_any(out.get("a", {"t": "const", "v": 0.0}), rng, cols, tier, 2)
        out["b"] = mutate_expr_any(out.get("b", {"t": "const", "v": 0.0}), rng, cols, tier, 2)
        return out

    if t == "log":
        if rng.random() < 0.25:
            out["op"] = rng.choice(LOG_OPS)
        out["a"] = mutate_bool_any(out.get("a", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), rng, cols, tier, max(0, max_depth - 1))
        out["b"] = mutate_bool_any(out.get("b", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), rng, cols, tier, max(0, max_depth - 1))
        if tier == "complex" and rng.random() < 0.10:
            out = {"t": "log", "op": rng.choice(LOG_OPS), "a": out, "b": random_bool_expr(rng, cols, 1, tier)}
        return out

    if t == "not":
        out["a"] = mutate_bool_any(out.get("a", {"t": "cmp", "op": "gt", "a": {"t": "const", "v": 1}, "b": {"t": "const", "v": 0}}), rng, cols, tier, max(0, max_depth - 1))
        return out

    return random_bool_expr(rng, cols, max_depth, tier)


def template_to_score_expr(template: Dict[str, Any]) -> Dict[str, Any]:
    kind = template.get("kind")
    if kind == "feature_rank":
        node = {"t": "col", "name": template["feature"]}
        if template.get("direction", "high") == "low":
            node = {"t": "u", "op": "neg", "a": node}
        return node

    if kind == "feature_spread":
        a = {"t": "col", "name": template["a"]}
        b = {"t": "col", "name": template["b"]}
        node = {"t": "b", "op": "sub", "a": a, "b": b}
        if template.get("direction", "high") == "low":
            node = {"t": "u", "op": "neg", "a": node}
        return node

    if kind == "feature_threshold":
        node = {"t": "col", "name": template["feature"]}
        transform = template.get("transform", "identity")
        if transform == "tanh":
            node = {"t": "u", "op": "tanh", "a": node}
        elif transform == "abs":
            node = {"t": "u", "op": "abs", "a": node}
        elif transform == "sqrt":
            node = {"t": "u", "op": "sqrt", "a": node}
        elif transform == "log1p":
            node = {"t": "u", "op": "log1p", "a": node}
        if template.get("direction", "high") == "low":
            node = {"t": "u", "op": "neg", "a": node}
        return node

    return {"t": "const", "v": 0.0}


def template_to_str(template: Optional[Dict[str, Any]]) -> str:
    if not template:
        return "template:none"
    kind = template.get("kind")
    if kind == "feature_rank":
        return f"rank({template.get('feature')},{template.get('direction','high')})"
    if kind == "feature_spread":
        core = f"({template.get('a')} - {template.get('b')})"
        return core if template.get("direction", "high") == "high" else f"neg({core})"
    if kind == "feature_threshold":
        feat = str(template.get("feature"))
        transform = template.get("transform", "identity")
        expr = feat if transform == "identity" else f"{transform}({feat})"
        return expr if template.get("direction", "high") == "high" else f"neg({expr})"
    return "template:unknown"


def random_template(rng: random.Random, cols: List[str], tier: str) -> Dict[str, Any]:
    cols = cols or ["close"]
    kinds = ["feature_rank", "feature_spread", "feature_threshold"]

    kind = rng.choice(kinds)
    if kind == "feature_rank":
        return {
            "kind": kind,
            "feature": rng.choice(cols),
            "direction": rng.choice(["high", "low"]),
        }
    if kind == "feature_spread":
        a = rng.choice(cols)
        b = rng.choice(cols) if cols else a
        if len(cols) > 1:
            while b == a:
                b = rng.choice(cols)
        return {
            "kind": kind,
            "a": a,
            "b": b,
            "direction": rng.choice(["high", "low"]),
        }
    return {
        "kind": "feature_threshold",
        "feature": rng.choice(cols),
        "direction": rng.choice(["high", "low"]),
        "transform": rng.choice(["identity", "tanh", "abs", "sqrt", "log1p"] if tier != "simple" else ["identity", "tanh"]),
    }


def mutate_template(template: Dict[str, Any], rng: random.Random, cols: List[str], tier: str) -> Dict[str, Any]:
    if rng.random() < 0.12:
        return random_template(rng, cols, tier)

    out = dict(template)
    kind = out.get("kind")
    cols = cols or ["close"]

    if kind == "feature_rank":
        if rng.random() < 0.60:
            out["feature"] = rng.choice(cols)
        if rng.random() < 0.35:
            out["direction"] = "low" if out.get("direction", "high") == "high" else "high"
        return out

    if kind == "feature_spread":
        if rng.random() < 0.50:
            out["a"] = rng.choice(cols)
        if rng.random() < 0.50:
            out["b"] = rng.choice(cols)
        if len(cols) > 1 and out.get("b") == out.get("a"):
            choices = [c for c in cols if c != out["a"]]
            out["b"] = rng.choice(choices)
        if rng.random() < 0.35:
            out["direction"] = "low" if out.get("direction", "high") == "high" else "high"
        return out

    if kind == "feature_threshold":
        if rng.random() < 0.60:
            out["feature"] = rng.choice(cols)
        if rng.random() < 0.35:
            out["direction"] = "low" if out.get("direction", "high") == "high" else "high"
        if rng.random() < 0.35:
            out["transform"] = rng.choice(["identity", "tanh", "abs", "sqrt", "log1p"] if tier != "simple" else ["identity", "tanh"])
        return out

    return random_template(rng, cols, tier)


# ============================================================
# Strategy genome
# ============================================================
@dataclass
class Genome:
    family: str = "expr"  # expr|template
    complexity_tier: str = "complex"  # simple|medium|complex
    score_expr: Dict[str, Any] = field(default_factory=lambda: {"t": "const", "v": 0.0})
    template: Optional[Dict[str, Any]] = None

    filters: List[Dict[str, Any]] = field(default_factory=list)

    select_mode: str = "threshold"  # threshold|topk|random
    thr: float = 0.0
    top_k: int = 1
    side_mode: str = "both"  # both|long_only|short_only
    horizon: int = 1
    cost_mult: float = 1.0
    random_trade_prob: float = 0.10

    def normalize(self) -> "Genome":
        self.family = self.family if self.family in VALID_FAMILIES else "expr"
        self.complexity_tier = self.complexity_tier if self.complexity_tier in VALID_TIERS else "complex"

        if self.family == "template":
            if not self.template or self.template.get("kind") not in VALID_TEMPLATE_KINDS:
                self.template = {"kind": "feature_rank", "feature": "close", "direction": "high"}
            self.score_expr = template_to_score_expr(self.template)

        self.select_mode = self.select_mode if self.select_mode in ("threshold", "topk", "random") else "threshold"
        self.side_mode = self.side_mode if self.side_mode in ("both", "long_only", "short_only") else "both"
        self.thr = clamp(float(self.thr), 0.0, 100.0)
        self.top_k = int(clamp(int(self.top_k), 1, 6))
        self.horizon = int(clamp(int(self.horizon), 1, 200))
        self.cost_mult = clamp(float(self.cost_mult), 0.0, 50.0)
        self.random_trade_prob = clamp(float(self.random_trade_prob), 0.0, 1.0)
        if self.filters is None:
            self.filters = []
        return self


def genome_id(g: Genome) -> str:
    payload = json.dumps(asdict(g.normalize()), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def genome_expr_str(g: Genome) -> str:
    if g.family == "template":
        return template_to_str(g.template)
    return expr_to_str(g.score_expr)


def genome_complexity(g: Genome) -> int:
    total = expr_node_count(g.score_expr)
    for f in g.filters:
        total += expr_node_count(f)
    total += len(g.filters) * 3
    if g.family == "template":
        total = max(3, total)
    return total


def random_select_mode(rng: random.Random, tier: str) -> str:
    return rng.choices(["threshold", "topk", "random"], weights=select_mode_weights_for_tier(tier), k=1)[0]


def random_filters(rng: random.Random, cols: List[str], tier: str, max_expr_depth: int) -> List[Dict[str, Any]]:
    max_filters = max_filters_for_tier(tier)
    if max_filters <= 0:
        return []
    if tier == "simple":
        n = rng.choices([0, 1], weights=[0.75, 0.25], k=1)[0]
    elif tier == "medium":
        n = rng.choices([0, 1, 2], weights=[0.45, 0.35, 0.20], k=1)[0]
    else:
        n = rng.randint(0, max_filters)
    return [random_bool_expr(rng, cols, max_bool_depth_for_tier(tier, max_expr_depth), tier) for _ in range(n)]


def random_genome(rng: random.Random, cols: List[str], tier: str, max_expr_depth: int) -> Genome:
    template_w, expr_w = family_weights_for_tier(tier)
    family = rng.choices(["template", "expr"], weights=[template_w, expr_w], k=1)[0]

    select_mode = random_select_mode(rng, tier)
    side_mode = rng.choice(["both", "long_only", "short_only"])

    if family == "template":
        template = random_template(rng, cols, tier)
        score_expr = template_to_score_expr(template)
        if template.get("kind") in ("feature_rank", "feature_spread") and rng.random() < 0.70:
            select_mode = "topk"
        elif template.get("kind") == "feature_threshold":
            select_mode = "threshold"

        if select_mode == "threshold":
            thr = clamp(abs(rng.gauss(0.4 if tier == "simple" else 0.8, 0.35)), 0.0, 5.0)
        else:
            thr = 0.0

        top_k_choices = [1, 2, 3, 5, 6]
        top_k = rng.choice(top_k_choices)
    else:
        depth = rng.randint(1, max_num_depth_for_tier(tier, max_expr_depth))
        score_expr = random_num_expr(rng, cols, depth=depth, tier=tier)
        template = None
        thr = abs(random_const(rng))
        top_k = rng.choice([1, 2, 3, 5, 6])

    g = Genome(
        family=family,
        complexity_tier=tier,
        template=template,
        score_expr=score_expr,
        filters=random_filters(rng, cols, tier, max_expr_depth),
        select_mode=select_mode,
        thr=thr,
        top_k=top_k,
        side_mode=side_mode,
        horizon=rng.choice([1, 2, 3, 6, 12] if tier != "simple" else [1, 2, 3, 6]),
        cost_mult=abs(rng.gauss(1.0, 0.35 if tier == "simple" else 0.45)),
        random_trade_prob=clamp(rng.random() * 0.35, 0.01, 0.35),
    )
    return g.normalize()


def mutate(g: Genome, rng: random.Random, cols: List[str], max_expr_depth: int) -> Genome:
    if rng.random() < 0.06:
        return random_genome(rng, cols, g.complexity_tier, max_expr_depth)

    h = Genome(**asdict(g)).normalize()
    tier = h.complexity_tier

    if rng.random() < 0.10:
        # family flip, same tier
        family = "expr" if h.family == "template" else "template"
        fresh = random_genome(rng, cols, tier, max_expr_depth)
        h.family = family
        if family == "template":
            h.template = random_template(rng, cols, tier)
            h.score_expr = template_to_score_expr(h.template)
        else:
            h.template = None
            h.score_expr = random_num_expr(rng, cols, max_num_depth_for_tier(tier, max_expr_depth), tier)
        h.select_mode = fresh.select_mode
        h.thr = fresh.thr
        h.top_k = fresh.top_k

    if h.family == "template":
        if rng.random() < 0.80:
            h.template = mutate_template(h.template or {"kind": "feature_rank", "feature": "close", "direction": "high"}, rng, cols, tier)
            h.score_expr = template_to_score_expr(h.template)
    else:
        if rng.random() < 0.80:
            h.score_expr = mutate_expr_any(h.score_expr, rng, cols, tier, max_num_depth_for_tier(tier, max_expr_depth))

    if rng.random() < 0.65:
        max_filters = max_filters_for_tier(tier)
        if rng.random() < 0.25 and len(h.filters) < max_filters:
            h.filters.append(random_bool_expr(rng, cols, max_bool_depth_for_tier(tier, max_expr_depth), tier))
        if rng.random() < 0.20 and len(h.filters) > 0:
            h.filters.pop(rng.randrange(len(h.filters)))
        for i in range(len(h.filters)):
            if rng.random() < 0.55:
                h.filters[i] = mutate_bool_any(h.filters[i], rng, cols, tier, max_bool_depth_for_tier(tier, max_expr_depth))

    if rng.random() < 0.20:
        h.select_mode = random_select_mode(rng, tier)
    if rng.random() < 0.45:
        h.thr = abs(h.thr + rng.gauss(0, 0.35 if tier == "simple" else 0.75))
    if rng.random() < 0.30:
        step_choices = [-3, -1, 1, 3] if tier == "simple" else [-5, -3, -1, 1, 3, 5]
        h.top_k = int(clamp(h.top_k + rng.choice(step_choices), 1, 6))
    if rng.random() < 0.12:
        h.side_mode = rng.choice(["both", "long_only", "short_only"])
    if rng.random() < 0.35:
        h.horizon = rng.choice([1, 2, 3, 4, 6, 8, 12, 16] if tier != "simple" else [1, 2, 3, 4, 6])
    if rng.random() < 0.35:
        h.cost_mult = abs(h.cost_mult + rng.gauss(0, 0.25 if tier == "simple" else 0.35))
    if rng.random() < 0.25:
        h.random_trade_prob = clamp(h.random_trade_prob + rng.gauss(0, 0.06), 0.0, 1.0)

    return h.normalize()


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    da = asdict(a)
    db = asdict(b)
    child = {}
    for k in da.keys():
        child[k] = da[k] if rng.random() < 0.5 else db[k]
    # keep tier from parent a to preserve bucket counts
    child["complexity_tier"] = a.complexity_tier
    if child.get("family") == "template" and child.get("template"):
        child["score_expr"] = template_to_score_expr(child["template"])
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
        first = values[0]
        if len(first) < len(header) or any(str(first[i]).strip() != header[i] for i in range(min(len(first), len(header)))):
            self._with_retry(lambda: ws.insert_row(header, 1, value_input_option="RAW"), what=f"insert_row {ws.title}")

    def append_strategies(self, header: List[str], rows: List[List[Any]]) -> None:
        ws = self._ws(self.strategies_tab, rows=20000, cols=max(60, len(header) + 2))
        self._ensure_header_row(ws, header)
        self._with_retry(lambda: ws.append_rows(rows, value_input_option="RAW"), what=f"append_rows {ws.title}")

    def write_champion(self, header: List[str], row: List[Any]) -> None:
        ws = self._ws(self.champion_tab, rows=2000, cols=max(60, len(header) + 2))
        self._with_retry(lambda: ws.clear(), what=f"clear {ws.title}")
        self._with_retry(lambda: ws.append_row(header, value_input_option="RAW"), what=f"append_row {ws.title}")
        self._with_retry(lambda: ws.append_row(row, value_input_option="RAW"), what=f"append_row {ws.title}")

    def replace_signals(self, header: List[str], rows: List[List[Any]]) -> None:
        ws = self._ws(self.signals_tab, rows=2000, cols=max(60, len(header) + 2))
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

    if "symbol" in d.columns:
        sym = d["symbol"]
        mask = sym.notna() & (sym.astype(str).str.strip() != "") & (sym.astype(str).str.lower() != "nan")
        d = d[mask].copy()
        d["symbol"] = d["symbol"].astype(str).str.strip()

    if "asof_utc" in d.columns:
        d["asof_utc"] = pd.to_datetime(d["asof_utc"], errors="coerce", utc=True)
        d = d[d["asof_utc"].notna()].copy()

    for c in d.columns:
        if c in ("symbol", "asof_utc", "status"):
            continue
        if c == "tradeable":
            continue
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


def _cap_random_choices_by_timestamp(tmp: pd.DataFrame, top_k: int, rng: np.random.Generator, side_mode: str, n_total: int) -> np.ndarray:
    sig = np.zeros(n_total, dtype=int)

    if "asof_utc" not in tmp.columns:
        groups = [(None, tmp)]
    else:
        groups = list(tmp.groupby("asof_utc", sort=False))

    for _, grp in groups:
        if grp.empty:
            continue
        idxs = grp["idx"].to_numpy(dtype=int)
        side = grp["side"].to_numpy(dtype=int)

        if side_mode == "both":
            long_idxs = idxs[side == 1]
            short_idxs = idxs[side == -1]
            if len(long_idxs) > top_k:
                long_idxs = rng.choice(long_idxs, size=top_k, replace=False)
            if len(short_idxs) > top_k:
                short_idxs = rng.choice(short_idxs, size=top_k, replace=False)
            sig[long_idxs] = 1
            sig[short_idxs] = -1
        else:
            chosen = idxs
            if len(chosen) > top_k:
                chosen = rng.choice(chosen, size=top_k, replace=False)
            sig[chosen] = 1 if side_mode == "long_only" else -1
    return sig


def make_signals_for_frame(d: pd.DataFrame, g: Genome, rng_seed: int) -> pd.Series:
    """
    Returns a Series of signals indexed like d:
      +1 long, -1 short, 0 none.
    """
    n = len(d)
    sig = np.zeros(n, dtype=int)

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

    if g.select_mode == "random":
        rng = np.random.default_rng(rng_seed)
        chosen = rng.random(n) < g.random_trade_prob
        chosen = chosen & ok

        side = rng.choice([-1, 1], size=n)
        if g.side_mode == "long_only":
            side = np.ones(n, dtype=int)
        elif g.side_mode == "short_only":
            side = -np.ones(n, dtype=int)

        tmp = d[["asof_utc"]].copy() if "asof_utc" in d.columns else pd.DataFrame(index=d.index)
        tmp["idx"] = np.arange(n)
        tmp["chosen"] = chosen
        tmp["side"] = side
        tmp = tmp[tmp["chosen"]].copy()
        if tmp.empty:
            return pd.Series(sig, index=d.index)

        capped = _cap_random_choices_by_timestamp(tmp, g.top_k, rng, g.side_mode, n)
        return pd.Series(capped, index=d.index)

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

    tmp = d[["asof_utc"]].copy() if "asof_utc" in d.columns else pd.DataFrame({"asof_utc": pd.Timestamp("1970-01-01", tz="UTC")}, index=d.index)
    tmp["score"] = score
    tmp["ok"] = ok
    tmp["idx"] = np.arange(len(tmp))

    sig = np.zeros(len(tmp), dtype=int)

    for _, grp in tmp.groupby("asof_utc", sort=False):
        gg = grp[grp["ok"] & np.isfinite(grp["score"])]
        if gg.empty:
            continue

        long_idx: set = set()
        if g.side_mode != "short_only":
            longs = gg.sort_values("score", ascending=False).head(g.top_k)
            long_idx = set(longs["idx"].tolist())
            sig[list(long_idx)] = 1

        if g.side_mode != "long_only":
            short_pool = gg if not long_idx else gg[~gg["idx"].isin(long_idx)]
            shorts = short_pool.sort_values("score", ascending=True).head(g.top_k)
            sig[shorts["idx"].to_numpy(dtype=int)] = -1

    return pd.Series(sig, index=d.index)


def trade_returns(df: pd.DataFrame, g: Genome, rng_seed: int) -> pd.DataFrame:
    h = g.horizon
    col = f"fwd_ret_{h}"
    if col not in df.columns:
        return df.iloc[0:0].copy()

    d = df.copy()
    d = d[d[col].notna()].copy()
    if d.empty:
        return d

    sig = make_signals_for_frame(d, g, rng_seed=rng_seed).to_numpy(dtype=int)
    d["signal"] = sig
    d = d[d["signal"] != 0].copy()
    if d.empty:
        return d

    spread = d["spread_bps"] if "spread_bps" in d.columns else 0.0
    spread = pd.to_numeric(spread, errors="coerce").fillna(0.0)
    cost = g.cost_mult * (spread / 10000.0)

    ret = d["signal"].astype(float) * d[col].astype(float) - cost.astype(float)
    out = d[["asof_utc", "symbol"]].copy()
    out["ret"] = ret.to_numpy(dtype=float)
    out["signal"] = d["signal"].to_numpy(dtype=int)
    out = out.sort_values(["asof_utc", "symbol"]).reset_index(drop=True)
    return out


def score_rets(rets: np.ndarray, min_trades: int) -> Dict[str, Any]:
    n = int(len(rets))
    if n == 0:
        return {
            "n": 0,
            "bar_n": 0,
            "trade_n": 0,
            "fitness": -1e12,
            "fitness_soft": -1e12,
            "fitness_hard": -1e12,
            "raw_fitness": -1e12,
            "total": 0.0,
            "sharpe": 0.0,
            "mdd": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "trade_scale": 0.0,
        }

    mean = float(np.mean(rets))
    std = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    sharpe = 0.0 if std == 0.0 else mean / std * math.sqrt(n)
    total = float(np.prod(1.0 + rets) - 1.0)
    mdd = max_drawdown(rets)

    raw_fitness = (2.0 * sharpe) + (12.0 * total) - (6.0 * mdd)
    trade_scale = min(1.0, n / max(min_trades, 1))
    fitness_soft = trade_scale * raw_fitness
    fitness_hard = raw_fitness if n >= min_trades else (-1e9 + n)

    return {
        "n": n,
        "bar_n": n,
        "trade_n": n,
        "fitness": float(fitness_soft),
        "fitness_soft": float(fitness_soft),
        "fitness_hard": float(fitness_hard),
        "raw_fitness": float(raw_fitness),
        "total": total,
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "mean": mean,
        "std": float(std),
        "trade_scale": float(trade_scale),
    }


def split_by_time(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q1 = df["asof_utc"].quantile(train_frac)
    q2 = df["asof_utc"].quantile(train_frac + val_frac)
    train = df[df["asof_utc"] <= q1]
    val = df[(df["asof_utc"] > q1) & (df["asof_utc"] <= q2)]
    test = df[df["asof_utc"] > q2]
    return train, val, test


def walk_forward_splits(df: pd.DataFrame, train_frac: float, val_frac: float, n_splits: int) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    ts = list(pd.Series(df["asof_utc"].dropna().unique()).sort_values())
    n_ts = len(ts)
    if n_ts < 6:
        return [split_by_time(df, train_frac, val_frac)]

    train_n = max(2, int(n_ts * train_frac))
    val_n = max(1, int(n_ts * val_frac))
    remaining = n_ts - train_n - val_n
    if remaining < 1:
        return [split_by_time(df, train_frac, val_frac)]

    step = max(1, remaining // max(1, n_splits))
    splits: List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

    for i in range(n_splits):
        train_end = min(train_n + i * step, n_ts - val_n - 1)
        val_end = min(train_end + val_n, n_ts - 1)
        test_end = min(val_end + step, n_ts)

        if train_end <= 0 or val_end <= train_end or test_end <= val_end:
            continue

        train_ts = set(ts[:train_end])
        val_ts = set(ts[train_end:val_end])
        test_ts = set(ts[val_end:test_end])

        if not val_ts or not test_ts:
            continue

        train = df[df["asof_utc"].isin(train_ts)]
        val = df[df["asof_utc"].isin(val_ts)]
        test = df[df["asof_utc"].isin(test_ts)]
        if not val.empty and not test.empty:
            splits.append((train, val, test))

    return splits or [split_by_time(df, train_frac, val_frac)]


def concat_bar_returns(frames: List[pd.DataFrame]) -> np.ndarray:
    """
    Convert trade-level returns into one equal-weight portfolio return per timestamp.
    This avoids scoring multiple trades from the same bar as independent observations.
    """
    if not frames:
        return np.array([], dtype=float)

    parts: List[np.ndarray] = []
    for f in frames:
        if f is None or f.empty:
            continue
        bars = (
            f.groupby("asof_utc", sort=True)["ret"]
            .mean()
            .to_numpy(dtype=float)
        )
        if len(bars):
            parts.append(bars)

    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def eval_genome(
    df: pd.DataFrame,
    g: Genome,
    cfg: "Config",
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    gid = genome_id(g)
    if cache is not None and gid in cache:
        return cache[gid]

    seed = int(gid, 16) % (2**31 - 1)

    if cfg.split_mode == "walk_forward":
        splits = walk_forward_splits(df, cfg.train_frac, cfg.val_frac, cfg.walk_forward_splits)
    else:
        splits = [split_by_time(df, cfg.train_frac, cfg.val_frac)]

    val_frames: List[pd.DataFrame] = []
    test_frames: List[pd.DataFrame] = []

    for i, (_, val, test) in enumerate(splits):
        tr_val = trade_returns(val, g, rng_seed=seed + i * 17)
        tr_test = trade_returns(test, g, rng_seed=seed + i * 17 + 7)
        if not tr_val.empty:
            val_frames.append(tr_val)
        if not tr_test.empty:
            test_frames.append(tr_test)

    val_rets = concat_bar_returns(val_frames)
    test_rets = concat_bar_returns(test_frames)

    val_score = score_rets(val_rets, min_trades=cfg.min_trades)
    test_score = score_rets(test_rets, min_trades=cfg.min_trades)

    val_score["trade_n"] = int(sum(len(f) for f in val_frames if f is not None))
    test_score["trade_n"] = int(sum(len(f) for f in test_frames if f is not None))

    complexity = genome_complexity(g)
    complexity_penalty = cfg.complexity_penalty * complexity
    consistency_penalty = cfg.consistency_penalty * abs(val_score["total"] - test_score["total"])

    val_selection_fitness = val_score["fitness_soft"] - complexity_penalty - consistency_penalty
    test_selection_fitness = test_score["fitness_soft"] - complexity_penalty - consistency_penalty

    reasons: List[str] = []
    if val_score["bar_n"] < cfg.champion_min_val_trades:
        reasons.append(f"val_bars<{cfg.champion_min_val_trades}")
    if test_score["bar_n"] < cfg.champion_min_test_trades:
        reasons.append(f"test_bars<{cfg.champion_min_test_trades}")
    if cfg.champion_require_positive_val_total and val_score["total"] <= 0:
        reasons.append("val_total<=0")
    if cfg.champion_require_positive_test_total and test_score["total"] <= 0:
        reasons.append("test_total<=0")
    if test_score["mdd"] > cfg.champion_max_test_mdd:
        reasons.append(f"test_mdd>{cfg.champion_max_test_mdd}")
    if g.complexity_tier not in cfg.allowed_live_tiers:
        reasons.append(f"tier_not_live:{g.complexity_tier}")

    champion_eligible = len(reasons) == 0

    res = {
        "val": val_score,
        "test": test_score,
        "complexity": complexity,
        "complexity_penalty": float(complexity_penalty),
        "consistency_penalty": float(consistency_penalty),
        "val_selection_fitness": float(val_selection_fitness),
        "test_selection_fitness": float(test_selection_fitness),
        "champion_eligible": champion_eligible,
        "champion_eligible_reasons": ";".join(reasons) if reasons else "ok",
    }
    if cache is not None:
        cache[gid] = res
    return res


# ============================================================
# Evolution helpers
# ============================================================
def tier_target_counts(cfg: "Config") -> Dict[str, int]:
    simple_n = int(round(cfg.pop_size * cfg.simple_frac))
    medium_n = int(round(cfg.pop_size * cfg.medium_frac))
    complex_n = cfg.pop_size - simple_n - medium_n
    if complex_n < 0:
        complex_n = 0
    total = simple_n + medium_n + complex_n
    while total < cfg.pop_size:
        complex_n += 1
        total += 1
    while total > cfg.pop_size and complex_n > 0:
        complex_n -= 1
        total -= 1
    return {"simple": simple_n, "medium": medium_n, "complex": complex_n}


def dedupe_population(pop: List[Genome]) -> List[Genome]:
    out: List[Genome] = []
    seen = set()
    for g in pop:
        gid = genome_id(g)
        if gid in seen:
            continue
        seen.add(gid)
        out.append(g)
    return out


def rebalance_population(pop: List[Genome], rng: random.Random, cols: List[str], cfg: "Config") -> List[Genome]:
    pop = dedupe_population(pop)

    targets = tier_target_counts(cfg)
    buckets: Dict[str, List[Genome]] = {tier: [] for tier in VALID_TIERS}
    leftovers: List[Genome] = []

    for g in pop:
        gg = Genome(**asdict(g)).normalize()
        if gg.complexity_tier in buckets:
            buckets[gg.complexity_tier].append(gg)
        else:
            leftovers.append(gg)

    out: List[Genome] = []
    for tier in VALID_TIERS:
        bucket = buckets[tier]
        target = targets[tier]
        if len(bucket) >= target:
            out.extend(bucket[:target])
            leftovers.extend(bucket[target:])
        else:
            out.extend(bucket)
            for _ in range(target - len(bucket)):
                if leftovers:
                    x = leftovers.pop()
                    x.complexity_tier = tier
                    out.append(mutate(x.normalize(), rng, cols, cfg.max_expr_depth))
                else:
                    out.append(random_genome(rng, cols, tier, cfg.max_expr_depth))
    return out[: cfg.pop_size]


def evaluate_population(
    df: pd.DataFrame,
    population: List[Genome],
    cfg: "Config",
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Tuple[Genome, Dict[str, Any]]]:
    return [(g, eval_genome(df, g, cfg, cache=cache)) for g in population]


def evolve_one_cycle(
    df: pd.DataFrame,
    population: List[Genome],
    rng: random.Random,
    cols: List[str],
    cfg: "Config",
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Genome]:
    pop = rebalance_population(population, rng, cols, cfg)

    for _ in range(cfg.generations_per_cycle):
        next_pop: List[Genome] = []

        for tier in VALID_TIERS:
            bucket = [g for g in pop if g.complexity_tier == tier]
            if not bucket:
                bucket = [random_genome(rng, cols, tier, cfg.max_expr_depth)]

            immigrants = max(1, int(len(bucket) * cfg.random_immigrant_rate))
            for _imm in range(immigrants):
                bucket[rng.randrange(len(bucket))] = random_genome(rng, cols, tier, cfg.max_expr_depth)

            scored = [(g, eval_genome(df, g, cfg, cache=cache)) for g in bucket]
            scored.sort(key=lambda x: x[1]["val_selection_fitness"], reverse=True)

            elite_n = max(2, int(len(bucket) * cfg.elite_frac))
            elite_n = min(elite_n, len(bucket))
            elites = [g for g, _ in scored[:elite_n]]

            bucket_next: List[Genome] = elites.copy()
            while len(bucket_next) < len(bucket):
                a = rng.choice(elites)
                b = rng.choice(elites)
                child = crossover(a, b, rng)
                child = mutate(child, rng, cols=cols, max_expr_depth=cfg.max_expr_depth)
                child.complexity_tier = tier
                bucket_next.append(child.normalize())

            next_pop.extend(bucket_next)

        next_pop = dedupe_population(next_pop)
        while len(next_pop) < cfg.pop_size:
            tier = rng.choice(list(VALID_TIERS))
            next_pop.append(random_genome(rng, cols, tier, cfg.max_expr_depth))

        pop = rebalance_population(next_pop, rng, cols, cfg)

    return pop


def choose_champion_candidate(scored: List[Tuple[Genome, Dict[str, Any]]]) -> Optional[Tuple[Genome, Dict[str, Any]]]:
    eligible = [(g, res) for g, res in scored if res["champion_eligible"]]
    if not eligible:
        return None
    return max(eligible, key=lambda x: x[1]["val_selection_fitness"])


# ============================================================
# Signals + champion logic
# ============================================================
def signal_live_allowed(g: Genome, res: Dict[str, Any], cfg: "Config") -> bool:
    if g.complexity_tier not in cfg.allowed_live_tiers:
        return False
    if cfg.signal_require_eligible and not res["champion_eligible"]:
        return False
    return True


def choose_signal_genome(
    champ: Optional[Genome],
    champ_res: Optional[Dict[str, Any]],
    best_val_g: Genome,
    best_val_res: Dict[str, Any],
    cfg: "Config",
) -> Tuple[Optional[Genome], Optional[Dict[str, Any]], str]:
    if cfg.signals_source == "champion":
        if champ is not None and champ_res is not None and signal_live_allowed(champ, champ_res, cfg):
            return champ, champ_res, "champion"
        if signal_live_allowed(best_val_g, best_val_res, cfg):
            return best_val_g, best_val_res, "best_val_fallback"
        return None, None, "none"
    if signal_live_allowed(best_val_g, best_val_res, cfg):
        return best_val_g, best_val_res, "best_val"
    if champ is not None and champ_res is not None and signal_live_allowed(champ, champ_res, cfg):
        return champ, champ_res, "champion_fallback"
    return None, None, "none"


def compute_signals(screener_df: pd.DataFrame, g: Genome) -> pd.DataFrame:
    if screener_df.empty:
        return pd.DataFrame()

    d = screener_df.copy()

    if "symbol" not in d.columns:
        for c in d.columns:
            if str(c).strip().lower() == "symbol":
                d.rename(columns={c: "symbol"}, inplace=True)
                break

    for c in d.columns:
        if c in ("symbol", "asof_utc", "status", "tradeable"):
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if "asof_utc" in d.columns:
        d["asof_utc"] = pd.to_datetime(d["asof_utc"], errors="coerce", utc=True)

    seed = int(genome_id(g), 16) % (2**31 - 1)
    sig = make_signals_for_frame(d, g, rng_seed=seed).to_numpy(dtype=int)
    d["signal"] = sig
    d = d[d["signal"] != 0].copy()
    if d.empty:
        return d

    d["side"] = np.where(d["signal"] == 1, "LONG", "SHORT")
    d["strategy_family"] = g.family
    d["complexity_tier"] = g.complexity_tier
    d["complexity_score"] = genome_complexity(g)
    d["score_expr"] = genome_expr_str(g)
    d["select_mode"] = g.select_mode
    d["thr"] = g.thr
    d["top_k"] = g.top_k
    d["horizon"] = g.horizon
    d["side_mode"] = g.side_mode

    cols = [
        c
        for c in [
            "symbol",
            "side",
            "signal",
            "close",
            "spread_bps",
            "asof_utc",
            "strategy_family",
            "complexity_tier",
            "complexity_score",
            "side_mode",
            "select_mode",
            "thr",
            "top_k",
            "horizon",
            "score_expr",
        ]
        if c in d.columns
    ]
    sort_cols = ["symbol"] if "symbol" in d.columns else d.columns.tolist()
    return d[cols].sort_values(sort_cols)


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
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


# ============================================================
# Config + main
# ============================================================
@dataclass
class Config:
    data_source: str
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

    signals_source: str

    split_mode: str  # quantile|walk_forward
    walk_forward_splits: int

    simple_frac: float
    medium_frac: float

    complexity_penalty: float
    consistency_penalty: float

    champion_min_val_trades: int
    champion_min_test_trades: int
    champion_require_positive_val_total: bool
    champion_require_positive_test_total: bool
    champion_max_test_mdd: float
    allowed_live_tiers: List[str]
    signal_require_eligible: bool


def load_config() -> Config:
    min_trades = env_int("MIN_TRADES", 25)
    simple_frac = env_float("SIMPLE_FRAC", 0.35)
    medium_frac = env_float("MEDIUM_FRAC", 0.35)
    simple_frac = clamp(simple_frac, 0.0, 1.0)
    medium_frac = clamp(medium_frac, 0.0, max(0.0, 1.0 - simple_frac))

    allowed_live_tiers = [t for t in env_csv("ALLOWED_LIVE_TIERS", "simple,medium") if t in VALID_TIERS]
    if not allowed_live_tiers:
        allowed_live_tiers = ["simple", "medium"]

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
        min_trades=min_trades,
        seed=env_int("SEED", 1337),
        state_path=env_str("STATE_PATH", "/data/state.json") or "/data/state.json",
        signals_source=(env_str("SIGNALS_SOURCE", "champion") or "champion").lower(),
        split_mode=(env_str("SPLIT_MODE", "walk_forward") or "walk_forward").lower(),
        walk_forward_splits=env_int("WALK_FORWARD_SPLITS", 3),
        simple_frac=simple_frac,
        medium_frac=medium_frac,
        complexity_penalty=env_float("COMPLEXITY_PENALTY", 0.02),
        consistency_penalty=env_float("CONSISTENCY_PENALTY", 50.0),
        champion_min_val_trades=env_int("CHAMPION_MIN_VAL_TRADES", min_trades),
        champion_min_test_trades=env_int("CHAMPION_MIN_TEST_TRADES", min_trades),
        champion_require_positive_val_total=env_bool("CHAMPION_REQUIRE_POSITIVE_VAL_TOTAL", True),
        champion_require_positive_test_total=env_bool("CHAMPION_REQUIRE_POSITIVE_TEST_TOTAL", True),
        champion_max_test_mdd=env_float("CHAMPION_MAX_TEST_MDD", 0.25),
        allowed_live_tiers=allowed_live_tiers,
        signal_require_eligible=env_bool("SIGNAL_REQUIRE_ELIGIBLE", True),
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
    cols: List[str] = []
    for c in df.columns:
        if c in ("symbol", "asof_utc", "status", "tradeable"):
            continue
        if str(c).startswith("fwd_ret_"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, int(0.01 * len(df))):
            cols.append(c)
    return cols


def genome_to_summary_row(
    g: Genome,
    res: Dict[str, Any],
) -> List[Any]:
    return [
        genome_id(g),
        g.family,
        g.complexity_tier,
        g.side_mode,
        genome_complexity(g),
        float(res["val_selection_fitness"]),
        float(res["val"]["fitness_hard"]),
        int(res["val"]["trade_n"]),
        float(res["val"]["total"]),
        float(res["val"]["sharpe"]),
        float(res["val"]["mdd"]),
        int(res["test"]["trade_n"]),
        float(res["test"]["total"]),
        float(res["test"]["sharpe"]),
        float(res["test"]["mdd"]),
        bool(res["champion_eligible"]),
        res["champion_eligible_reasons"],
        genome_expr_str(g),
    ]


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit.")
    args = parser.parse_args()

    cfg = load_config()
    rng = random.Random(cfg.seed)
    client = make_client(cfg)
    state = load_state(cfg.state_path)

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

    feature_cols = extract_feature_cols(hist)
    if not feature_cols:
        raise RuntimeError("No usable numeric feature columns found in history.")

    pop: List[Genome] = []
    if "population" in state:
        try:
            pop = [Genome(**p).normalize() for p in state["population"]]
        except Exception:
            pop = []
    if not pop or len(pop) != cfg.pop_size:
        targets = tier_target_counts(cfg)
        pop = []
        for tier in VALID_TIERS:
            for _ in range(targets[tier]):
                pop.append(random_genome(rng, feature_cols, tier, cfg.max_expr_depth))

    champ: Optional[Genome] = None
    champ_score = -1e18
    if "champion" in state:
        try:
            champ = Genome(**state["champion"]["genome"]).normalize()
            champ_score = float(state["champion"]["score"])
        except Exception:
            champ = None
            champ_score = -1e18

    strategies_header = [
        "ts_utc",
        "cycle",
        "best_val_strategy_id",
        "best_val_family",
        "best_val_tier",
        "best_val_side_mode",
        "best_val_complexity",
        "best_val_selection_fitness",
        "best_val_val_fitness_hard",
        "best_val_val_trades",
        "best_val_val_total",
        "best_val_val_sharpe",
        "best_val_val_mdd",
        "best_val_test_trades",
        "best_val_test_total",
        "best_val_test_sharpe",
        "best_val_test_mdd",
        "best_val_eligible",
        "best_val_eligible_reasons",
        "best_val_expr",
        "best_test_strategy_id",
        "best_test_family",
        "best_test_tier",
        "best_test_side_mode",
        "best_test_complexity",
        "best_test_selection_fitness",
        "best_test_test_fitness_hard",
        "best_test_val_trades",
        "best_test_val_total",
        "best_test_test_trades",
        "best_test_test_total",
        "best_test_test_sharpe",
        "best_test_test_mdd",
        "best_test_eligible",
        "best_test_eligible_reasons",
        "best_test_expr",
        "champion_strategy_id",
        "champion_family",
        "champion_tier",
        "champion_side_mode",
        "champion_complexity",
        "champion_selection_fitness",
        "champion_val_trades",
        "champion_val_total",
        "champion_test_trades",
        "champion_test_total",
        "champion_test_sharpe",
        "champion_test_mdd",
        "champion_eligible",
        "champion_eligible_reasons",
        "champion_expr",
        "params_json",
    ]

    champion_header = [
        "ts_utc",
        "champion_strategy_id",
        "family",
        "complexity_tier",
        "side_mode",
        "complexity_score",
        "champion_selection_fitness",
        "champion_val_fitness_hard",
        "champion_val_trades",
        "champion_val_total",
        "champion_val_sharpe",
        "champion_val_mdd",
        "champion_test_fitness_hard",
        "champion_test_trades",
        "champion_test_total",
        "champion_test_sharpe",
        "champion_test_mdd",
        "champion_eligible",
        "champion_eligible_reasons",
        "select_mode",
        "thr",
        "top_k",
        "horizon",
        "filters_count",
        "score_expr",
        "params_json",
    ]

    cycle = int(state.get("cycle", 0))

    log.info(
        "Starting evolver v3: pop_size=%s features=%s split_mode=%s train/val/test=%s/%s/%s tiers=%s/%s/%s live_tiers=%s",
        cfg.pop_size,
        len(feature_cols),
        cfg.split_mode,
        cfg.train_frac,
        cfg.val_frac,
        1.0 - cfg.train_frac - cfg.val_frac,
        cfg.simple_frac,
        cfg.medium_frac,
        1.0 - cfg.simple_frac - cfg.medium_frac,
        ",".join(cfg.allowed_live_tiers),
    )

    while True:
        hist_raw = client.read_history(cfg.history_lookback_rows)
        hist = coerce_types(hist_raw)
        if hist.empty:
            log.warning("History empty; sleeping %ss", cfg.sleep_seconds)
            if args.once:
                return
            time.sleep(cfg.sleep_seconds)
            continue

        feature_cols = extract_feature_cols(hist) or feature_cols
        pop = rebalance_population(pop, rng, feature_cols, cfg)

        horizons = sorted(set([g.horizon for g in pop] + ([champ.horizon] if champ else []) + [1, 2, 3, 6, 12]))
        hist2 = add_forward_returns(hist, horizons)

        eval_cache: Dict[str, Dict[str, Any]] = {}
        pop = evolve_one_cycle(hist2, pop, rng, feature_cols, cfg, cache=eval_cache)
        scored = evaluate_population(hist2, pop, cfg, cache=eval_cache)

        best_val_g, best_val_res = max(scored, key=lambda x: x[1]["val_selection_fitness"])
        best_test_g, best_test_res = max(scored, key=lambda x: x[1]["test_selection_fitness"])

        candidate = choose_champion_candidate(scored)
        improved = False
        if candidate is not None:
            candidate_g, candidate_res = candidate
            if candidate_res["val_selection_fitness"] > champ_score:
                champ = candidate_g
                champ_score = float(candidate_res["val_selection_fitness"])
                improved = True

        champ_id = genome_id(champ) if champ else ""
        champ_res = eval_genome(hist2, champ, cfg, cache=eval_cache) if champ else None
        champ_expr = genome_expr_str(champ) if champ else ""

        cycle += 1
        state["cycle"] = cycle
        state["population"] = [asdict(g) for g in pop]
        if champ:
            state["champion"] = {"genome": asdict(champ), "score": champ_score}
        save_state(cfg.state_path, state)

        cycle_ts = now_utc_iso()

        params_json = json.dumps(
            {
                "best_val": asdict(best_val_g),
                "best_test": asdict(best_test_g),
                "champion": asdict(champ) if champ else None,
                "config": {
                    "split_mode": cfg.split_mode,
                    "walk_forward_splits": cfg.walk_forward_splits,
                    "allowed_live_tiers": cfg.allowed_live_tiers,
                },
            },
            sort_keys=True,
        )

        best_val_vals = genome_to_summary_row(best_val_g, best_val_res)

        row = [
            cycle_ts,
            cycle,
            *best_val_vals,
            genome_id(best_test_g),
            best_test_g.family,
            best_test_g.complexity_tier,
            best_test_g.side_mode,
            genome_complexity(best_test_g),
            float(best_test_res["test_selection_fitness"]),
            float(best_test_res["test"]["fitness_hard"]),
            int(best_test_res["val"]["trade_n"]),
            float(best_test_res["val"]["total"]),
            int(best_test_res["test"]["trade_n"]),
            float(best_test_res["test"]["total"]),
            float(best_test_res["test"]["sharpe"]),
            float(best_test_res["test"]["mdd"]),
            bool(best_test_res["champion_eligible"]),
            best_test_res["champion_eligible_reasons"],
            genome_expr_str(best_test_g),
            champ_id,
            champ.family if champ else "",
            champ.complexity_tier if champ else "",
            champ.side_mode if champ else "",
            genome_complexity(champ) if champ else "",
            float(champ_res["val_selection_fitness"]) if champ_res else "",
            int(champ_res["val"]["trade_n"]) if champ_res else "",
            float(champ_res["val"]["total"]) if champ_res else "",
            int(champ_res["test"]["trade_n"]) if champ_res else "",
            float(champ_res["test"]["total"]) if champ_res else "",
            float(champ_res["test"]["sharpe"]) if champ_res else "",
            float(champ_res["test"]["mdd"]) if champ_res else "",
            bool(champ_res["champion_eligible"]) if champ_res else "",
            champ_res["champion_eligible_reasons"] if champ_res else "",
            champ_expr,
            params_json,
        ]
        client.append_strategies(strategies_header, [row])

        if improved and champ and champ_res:
            champ_row = [
                cycle_ts,
                champ_id,
                champ.family,
                champ.complexity_tier,
                champ.side_mode,
                genome_complexity(champ),
                float(champ_res["val_selection_fitness"]),
                float(champ_res["val"]["fitness_hard"]),
                int(champ_res["val"]["trade_n"]),
                float(champ_res["val"]["total"]),
                float(champ_res["val"]["sharpe"]),
                float(champ_res["val"]["mdd"]),
                float(champ_res["test"]["fitness_hard"]),
                int(champ_res["test"]["trade_n"]),
                float(champ_res["test"]["total"]),
                float(champ_res["test"]["sharpe"]),
                float(champ_res["test"]["mdd"]),
                bool(champ_res["champion_eligible"]),
                champ_res["champion_eligible_reasons"],
                champ.select_mode,
                champ.thr,
                champ.top_k,
                champ.horizon,
                len(champ.filters),
                champ_expr,
                json.dumps(asdict(champ), sort_keys=True),
            ]
            client.write_champion(champion_header, champ_row)

        signal_g, _signal_res, signal_source = choose_signal_genome(champ, champ_res, best_val_g, best_val_res, cfg)
        screener = client.read_screener()

        if signal_g is None:
            signals = pd.DataFrame()
        else:
            signals = compute_signals(screener, signal_g)

        header = ["ts_utc", "strategy_id", "source"] + list(signals.columns)
        rows: List[List[Any]] = []
        sid = genome_id(signal_g) if signal_g else ""
        for _, r in signals.iterrows():
            rows.append([cycle_ts, sid, signal_source] + [r.get(c, "") for c in signals.columns])
        client.replace_signals(header, rows)

        log.info(
            "cycle=%s best_val=%s val_sel=%.3f best_test=%s test_sel=%.3f champion=%s champ_val_sel=%.3f improved=%s signal_source=%s",
            cycle,
            genome_id(best_val_g),
            float(best_val_res["val_selection_fitness"]),
            genome_id(best_test_g),
            float(best_test_res["test_selection_fitness"]),
            champ_id,
            float(champ_res["val_selection_fitness"]) if champ_res else -1e18,
            improved,
            signal_source,
        )

        if args.once:
            return

        time.sleep(cfg.sleep_seconds)


if __name__ == "__main__":
    main()
