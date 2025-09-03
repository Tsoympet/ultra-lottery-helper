#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Lottery Helper — Offline Suite (v6.3.0, Colab-Compatible)
- Offline by default; optional online history fetching
- History auto-merge per game (CSV/XLS/XLSX) from data/history/<game>/
- EWMA + BMA + Luck/Unluck (adaptive) + ML ensemble (Prophet, LightGBM, RF, XGBoost, SVM)
- Dynamic ensemble weighting via CV performance
- Feature engineering (sums, odd/even, pairs, gaps, clusters)
- Adaptive constraints based on historical distributions
- Gumbel Top-k sampler + constraints (+ optional wheels)
- DPP/Greedy portfolio selector with Monte Carlo risk assessment and coverage optimization
- Walk-forward cross-validation + self-learning replay with online luck/unluck updates
- Optional EV re-rank (cost-aware) with OPAP-style defaults
- Diagnostics (frequency/recency/last-digit/pairs/odd-even)
- Exports CSV/PNG (6 columns) into ./exports/<game>/
- Plot caching per (game, history signature)
- Debounce-style UI feedback for heavy sliders
"""

import os
import math
import glob
import itertools
import json
import textwrap
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

# Optional ML libraries (safe imports)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.model_selection import GridSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.svm import SVC
    SVM_AVAILABLE = True
except ImportError:
    SVM_AVAILABLE = False

# Optional online fetch
import requests

# =============================================================================
# Paths & small utils
# =============================================================================

def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

# Colab-compatible paths
ROOT = "/content" if _in_colab() else os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(ROOT, "data", "history")
EXPORT_ROOT = os.path.join(ROOT, "exports")
os.makedirs(EXPORT_ROOT, exist_ok=True)

def _rng(seed: Optional[int]):
    return np.random.default_rng(int(seed or 42))

def _now_ts():
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_game_export_dir(game: str):
    path = os.path.join(EXPORT_ROOT, game.lower())
    os.makedirs(path, exist_ok=True)
    return path

def _quick_df_sig(df: pd.DataFrame, game: str) -> str:
    """Signature of history to drive plot cache."""
    if df is None or len(df) == 0:
        return f"{game}|empty"
    cols = [c for c in df.columns if c in ("n1","n2","n3","n4","n5","n6","e1","e2","joker","date")]
    if not cols:
        cols = list(df.columns)
    sample = df[cols].tail(100)
    buf = sample.to_csv(index=False).encode("utf-8")
    h = hashlib.sha1(buf).hexdigest()
    return f"{game}|{len(df)}|{h}"

# =============================================================================
# Game specs + OPAP-style price defaults
# =============================================================================

@dataclass
class GameSpec:
    name: str
    main_pick: int
    main_max: int
    sec_pick: int
    sec_max: int
    cols: List[str]

GAMES: Dict[str, GameSpec] = {
    "TZOKER": GameSpec("TZOKER", 5, 45, 1, 20, ["n1","n2","n3","n4","n5","joker"]),
    "LOTTO":  GameSpec("LOTTO",  6, 49, 0,  0, ["n1","n2","n3","n4","n5","n6"]),
    "EUROJACKPOT": GameSpec("EUROJACKPOT", 5, 50, 2, 12, ["n1","n2","n3","n4","n5","e1","e2"]),
}

OPAP_TICKET_PRICE_DEFAULTS = {
    "TZOKER":      0.50,
    "LOTTO":       0.50,
    "EUROJACKPOT": 2.00,
}

def _game_path(game: str) -> str:
    return os.path.join(DATA_ROOT, game.lower())

# =============================================================================
# Data Loading (Offline + Optional Online)
# =============================================================================

def fetch_online_history(game: str) -> Tuple[pd.DataFrame, str]:
    urls = {
        "TZOKER": "https://www.opap.gr/en/web/opap-gr/tzoker-draw-results",
        "LOTTO": "https://www.opap.gr/en/web/opap-gr/lotto-draw-results",
        "EUROJACKPOT": "https://www.eurojackpot.org/en/results/"
    }
    try:
        response = requests.get(urls[game], timeout=10)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        if not tables:
            return pd.DataFrame(), f"No HTML tables found for {game} online."
        df = tables[0]
        df.columns = [str(c).strip().lower() for c in df.columns]
        spec = GAMES[game]
        needed = spec.cols
        col_map = {}
        for col in needed + ["date"]:
            cand = [c for c in df.columns if col in c]
            col_map[col] = cand[0] if cand else None
        missing = [k for k in needed if col_map.get(k) is None]
        if missing:
            return pd.DataFrame(), f"Online data missing columns: {missing}"
        out = pd.DataFrame({k: pd.to_numeric(df[col_map[k]], errors="coerce") for k in needed})
        out["date"] = pd.to_datetime(df[col_map["date"]], errors="coerce") if col_map.get("date") else pd.NaT
        out = out.dropna(how="any")
        for k in needed:
            out[k] = out[k].astype(int, errors="ignore")
        return out, f"Fetched {len(out)} draws from {urls[game]}"
    except Exception as e:
        return pd.DataFrame(), f"Failed to fetch online data for {game}: {str(e)}"

def _load_all_history(game: str, use_online: bool = False) -> Tuple[pd.DataFrame, str]:
    path = _game_path(game)
    files = sorted(
        glob.glob(os.path.join(path, "*.csv")) +
        glob.glob(os.path.join(path, "*.xlsx")) +
        glob.glob(os.path.join(path, "*.xls"))
    )
    frames = []
    skipped = []
    spec = GAMES[game]
    needed = spec.cols

    if use_online:
        online_df, online_log = fetch_online_history(game)
        if not online_df.empty:
            frames.append(online_df)
        skipped.append(online_log)

    for f in files:
        try:
            if f.lower().endswith(".csv"): df = pd.read_csv(f)
            else: df = pd.read_excel(f)
        except Exception as e:
            skipped.append(f"Skipped {os.path.basename(f)}: Failed to read ({str(e)})")
            continue
        df.columns = [str(c).strip().lower() for c in df.columns]
        col_map = {}
        for col in needed + ["date"]:
            cand = [c for c in df.columns if c == col]
            if not cand and col in ["e1","e2","joker"]:
                if col == "e1":
                    cand = [c for c in df.columns if c in ["euro1","euro_1","star1","e-1","euroa"]]
                if col == "e2":
                    cand = [c for c in df.columns if c in ["euro2","euro_2","star2","e-2","eurob"]]
                if col == "joker":
                    cand = [c for c in df.columns if c in ["j","tzoker","bonus"]]
            col_map[col] = cand[0] if cand else None
        missing = [k for k in needed if col_map.get(k) is None]
        if missing:
            skipped.append(f"Skipped {os.path.basename(f)}: Missing columns {missing}")
            continue
        out = pd.DataFrame({k: pd.to_numeric(df[col_map[k]], errors="coerce") for k in needed})
        out["date"] = pd.to_datetime(df[col_map["date"]], errors="coerce") if col_map.get("date") else pd.NaT
        out = out.dropna(how="any")
        for k in needed:
            out[k] = out[k].astype(int, errors="ignore")
        frames.append(out)

    if not frames:
        return pd.DataFrame(), f"No valid data loaded. Issues: {', '.join(skipped)}"

    df = pd.concat(frames, ignore_index=True)
    if game == "LOTTO":
        arr = np.sort(df[["n1","n2","n3","n4","n5","n6"]].to_numpy(), axis=1)
        df[["n1","n2","n3","n4","n5","n6"]] = arr
    else:
        arr = np.sort(df[["n1","n2","n3","n4","n5"]].to_numpy(), axis=1)
        df[["n1","n2","n3","n4","n5"]] = arr

    ok = np.ones(len(df), dtype=bool)
    for c in ["n1","n2","n3","n4","n5"]:
        ok &= df[c].between(1, spec.main_max, inclusive="both")
    if spec.name == "LOTTO":
        ok &= df["n6"].between(1, spec.main_max, inclusive="both")
    if spec.sec_pick == 1 and "joker" in df.columns:
        ok &= df["joker"].between(1, spec.sec_max, inclusive="both")
    if spec.sec_pick == 2 and {"e1","e2"}.issubset(df.columns):
        ok &= df["e1"].between(1, spec.sec_max, inclusive="both")
        ok &= df["e2"].between(1, spec.sec_max, inclusive="both")

    df = df[ok].copy()
    if "date" in df.columns:
        df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df, f"Loaded {len(df)} valid draws. Skipped: {', '.join(skipped) if skipped else 'None'}"

# =============================================================================
# Modeling (EWMA/BMA + Luck/Unluck + ML Ensemble)
# =============================================================================

@dataclass
class Config:
    iterations: int = 50000
    topk: int = 200
    seed: int = 42
    use_bma: bool = True
    bma_w_freq: float = 0.5
    bma_w_rec: float = 0.3
    bma_w_ml: float = 0.2
    use_ewma: bool = True
    half_life: int = 120
    use_ml: bool = False
    luck_beta: float = 0.10
    unluck_gamma: float = 0.05
    min_even: int = 0
    max_even: int = 6
    min_odd: int = 0
    max_odd: int = 6
    min_low: int = 0
    max_low: int = 6
    max_same_lastdigit: int = 3
    sum_min: int = 50
    sum_max: int = 240
    max_consecutive: int = 3
    optimizer: str = "DPP"
    portfolio_size: int = 6
    use_wheels: bool = False
    wheel_keys: str = ""
    enable_ev: bool = False
    ev_weight: float = 1.0
    ev_ticket_price: float = 2.0
    ev_tiers_json: str = "[]"
    use_online: bool = False
    monte_sims: int = 10000

    def set_adaptive_constraints(self, df: pd.DataFrame, spec: GameSpec):
        if len(df) < 30: return
        mains = df[spec.cols[:spec.main_pick]].values
        odds_counts = np.sum(mains % 2 == 1, axis=1)
        lows_counts = np.sum(mains <= spec.main_max // 2, axis=1)
        sums = np.sum(mains, axis=1)
        consec = []
        for row in mains:
            row = np.sort(row)
            curr = 1; best = 1
            for i in range(1, len(row)):
                if row[i] == row[i-1] + 1:
                    curr += 1; best = max(best, curr)
                else:
                    curr = 1
            consec.append(best)
        self.min_even = int(np.percentile(spec.main_pick - odds_counts, 10))
        self.max_even = int(np.percentile(spec.main_pick - odds_counts, 90))
        self.min_odd  = int(np.percentile(odds_counts, 10))
        self.max_odd  = int(np.percentile(odds_counts, 90))
        self.min_low  = int(np.percentile(lows_counts, 10))
        self.max_low  = int(np.percentile(lows_counts, 90))
        self.sum_min  = int(np.percentile(sums, 10))
        self.sum_max  = int(np.percentile(sums, 90))
        self.max_consecutive = int(np.percentile(consec, 90))

def _ewma_weights(n: int, half_life: int) -> np.ndarray:
    if n <= 0: return np.array([])
    lam = math.log(2.0) / max(1, half_life)
    idx = np.arange(n)[::-1]
    w = np.exp(-lam * idx)
    return w / w.sum()

def _luck_vectors(df: pd.DataFrame, game: str, spec: GameSpec) -> Tuple[np.ndarray, np.ndarray]:
    main_drought = np.zeros(spec.main_max, dtype=float)
    main_recent = np.zeros(spec.main_max, dtype=float)
    seen_last = {i: None for i in range(1, spec.main_max+1)}
    for idx, row in df.reset_index(drop=True).iterrows():
        present = set([int(row[c]) for c in ["n1","n2","n3","n4","n5"]])
        if spec.name == "LOTTO":
            present.add(int(row["n6"]))
        for n in range(1, spec.main_max+1):
            last = seen_last[n]
            if last is not None:
                gap = idx - last
                if gap > main_drought[n-1]:
                    main_drought[n-1] = gap
        for n in present:
            main_recent[n-1] += 1.0
            seen_last[n] = idx
    if main_drought.max() > 0: main_drought /= main_drought.max()
    if main_recent.max() > 0: main_recent /= main_recent.max()
    return main_drought, main_recent

def ml_probs(df: pd.DataFrame, game: str, cfg: Config, main_max: int) -> Optional[np.ndarray]:
    if not cfg.use_ml or len(df) < 120 or not SKLEARN_AVAILABLE:
        return None

    spec = GAMES[game]
    cols = ["n1","n2","n3","n4","n5"] + (["n6"] if spec.name == "LOTTO" else [])
    pick_size = len(cols)

    # ---------- Feature engineering per row ----------
    features = []
    per_row_labels: List[List[int]] = []
    for i in range(len(df)):
        row = df.iloc[i][cols].values.astype(int)
        row_sorted = np.sort(row)
        sums = int(np.sum(row_sorted))
        odds = int(np.sum(row_sorted % 2 == 1))
        gaps = float(np.mean(np.diff(row_sorted))) if pick_size > 1 else 0.0
        try:
            km = KMeans(n_clusters=min(3, pick_size), random_state=cfg.seed, n_init=5)
            clusters = km.fit_predict(row_sorted.reshape(-1, 1))
            cluster_counts = np.bincount(clusters, minlength=min(3, pick_size)).tolist()
            if len(cluster_counts) < 3:
                cluster_counts += [0] * (3 - len(cluster_counts))
        except Exception:
            cluster_counts = [pick_size, 0, 0]

        pairs = int(len(list(itertools.combinations(row_sorted, 2))))
        features.append([sums, odds, pairs, gaps] + cluster_counts[:3])
        per_row_labels.append((row_sorted - 1).tolist())

    X_rows = np.array(features, dtype=float)

    # ---------- Expand to one sample per number ----------
    X_rep_list, y_rep_list = [], []
    for i in range(len(X_rows)):
        for cls in per_row_labels[i]:
            X_rep_list.append(X_rows[i])
            y_rep_list.append(int(cls))
    if not X_rep_list:
        return None
    X_rep = np.vstack(X_rep_list)
    y_rep = np.array(y_rep_list, dtype=int)

    models, scores = [], []

    def fit_and_maybe_keep(name: str, grid):
        try:
            grid.fit(X_rep, y_rep)
            score = float(getattr(grid, "best_score_", 0.0))
            est = getattr(grid, "best_estimator_", None)
            if est is not None and score > 0.1:
                models.append((name, est, score))
                scores.append(score)
        except Exception:
            pass

    if LIGHTGBM_AVAILABLE:
        fit_and_maybe_keep(
            "lightgbm",
            GridSearchCV(
                lgb.LGBMClassifier(random_state=cfg.seed),
                {"n_estimators": [50, 100, 200], "max_depth": [3, 5]},
                cv=3, n_jobs=1,
            )
        )
    if SKLEARN_AVAILABLE:
        fit_and_maybe_keep(
            "rf",
            GridSearchCV(
                RandomForestClassifier(random_state=cfg.seed),
                {"n_estimators": [50, 100], "max_depth": [3, 5]},
                cv=3, n_jobs=1,
            )
        )
    if XGBOOST_AVAILABLE:
        fit_and_maybe_keep(
            "xgboost",
            GridSearchCV(
                xgb.XGBClassifier(random_state=cfg.seed, use_label_encoder=False, eval_metric="mlogloss"),
                {"n_estimators": [50, 100], "max_depth": [3, 5]},
                cv=3, n_jobs=1,
            )
        )
    if SVM_AVAILABLE:
        fit_and_maybe_keep(
            "svm",
            GridSearchCV(
                SVC(probability=True, random_state=cfg.seed),
                {"C": [0.1, 1.0], "kernel": ["rbf"]},
                cv=3, n_jobs=1,
            )
        )

    prophet_probs = None
    if PROPHET_AVAILABLE and "date" in df.columns and df["date"].notna().sum() >= 120:
        try:
            prophet_probs = np.zeros(main_max, dtype=float)
            for num in range(1, main_max + 1):
                ts_data = df[["date"]].copy()
                ts_data["y"] = df[cols].apply(lambda row: 1.0 if num in row.values else 0.0, axis=1)
                ts_data = ts_data[["date", "y"]].rename(columns={"date": "ds"})
                ts_data = ts_data[ts_data["ds"].notna()]
                if len(ts_data) < 120:
                    continue
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    n_changepoints=10,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                )
                model.fit(ts_data)
                future = model.make_future_dataframe(periods=1, freq="D")
                forecast = model.predict(future)
                prophet_probs[num - 1] = max(0.0, float(forecast["yhat"].iloc[-1]))
            if prophet_probs.sum() > 0:
                prophet_probs /= prophet_probs.sum()
            else:
                prophet_probs = None
        except Exception:
            prophet_probs = None

    if not models and prophet_probs is None:
        return None

    latest_feats = X_rows[-1].reshape(1, -1)
    scores_arr = np.array(scores, dtype=float) if scores else np.array([1.0])
    weights = scores_arr / scores_arr.sum() if scores else np.array([1.0])

    blended = np.zeros(main_max, dtype=float)
    for (name, est, w) in zip([m[0] for m in models], [m[1] for m in models], weights):
        try:
            proba = est.predict_proba(latest_feats)[0]
            classes = est.classes_
            tmp = np.zeros(main_max, dtype=float)
            for p, cls in zip(proba, classes):
                if 0 <= int(cls) < main_max:
                    tmp[int(cls)] = float(p)
            if tmp.sum() > 0:
                tmp = tmp / tmp.sum()
            blended += w * tmp
        except Exception:
            continue

    if prophet_probs is not None:
        blended = (blended + prophet_probs) / 2 if blended.sum() > 0 else prophet_probs
    if blended.sum() == 0:
        return None
    blended /= blended.sum()
    return blended

def build_probs(df: pd.DataFrame, game: str, cfg: Config) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    spec = GAMES[game]
    cfg.set_adaptive_constraints(df, spec)

    if len(df) == 0:
        return np.ones(spec.main_max)/spec.main_max, (np.ones(spec.sec_max)/spec.sec_max if spec.sec_pick>0 else None)

    main_counts = np.zeros(spec.main_max, dtype=float)
    cols = ["n1","n2","n3","n4","n5"] + (["n6"] if spec.name=="LOTTO" else [])
    for _, row in df.iterrows():
        for c in cols:
            main_counts[int(row[c])-1] += 1.0
    main_freq = (main_counts + 1e-3) / (main_counts.sum() + spec.main_max * 1e-3)

    rdf = df.tail(min(100, len(df)))
    rec_counts = np.zeros_like(main_counts)
    for _, row in rdf.iterrows():
        for c in cols:
            rec_counts[int(row[c])-1] += 1.0
    rec = (rec_counts + 1e-3) / (rec_counts.sum() + spec.main_max * 1e-3)

    if cfg.use_ewma:
        w = _ewma_weights(len(df), cfg.half_life)
        ew = np.zeros_like(main_counts)
        for i, (_, row) in enumerate(df.iterrows()):
            for c in cols:
                ew[int(row[c])-1] += w[i]
        ew = (ew + 1e-6) / (ew.sum() + spec.main_max * 1e-6)
    else:
        ew = main_freq

    ml_p = ml_probs(df, game, cfg, spec.main_max) if cfg.use_ml else None

    if cfg.use_bma:
        main_probs = cfg.bma_w_freq * main_freq + cfg.bma_w_rec * ((rec + ew)/2.0)
        if ml_p is not None:
            main_probs = (main_probs + cfg.bma_w_ml * ml_p) / (1.0 + cfg.bma_w_ml)
        main_probs = main_probs / main_probs.sum()
    else:
        main_probs = ew if ml_p is None else (ew + ml_p) / 2

    sec_probs = None
    if spec.sec_pick == 1 and "joker" in df.columns:
        sec_counts = np.zeros(spec.sec_max, dtype=float)
        for _, row in df.iterrows():
            sec_counts[int(row["joker"])-1] += 1.0
        sec_probs = (sec_counts + 1e-3) / (sec_counts.sum() + spec.sec_max * 1e-3)
    elif spec.sec_pick == 2 and {"e1","e2"}.issubset(df.columns):
        sec_counts = np.zeros(spec.sec_max, dtype=float)
        for _, row in df.iterrows():
            sec_counts[int(row["e1"])-1] += 1.0
            sec_counts[int(row["e2"])-1] += 1.0
        sec_probs = (sec_counts + 1e-3) / (sec_counts.sum() + spec.sec_max * 1e-3)

    drought, recent = _luck_vectors(df, game, spec)
    main_probs = main_probs * (1.0 + cfg.luck_beta * drought) * (1.0 - cfg.unluck_gamma * recent)
    main_probs = np.clip(main_probs, 1e-12, None); main_probs /= main_probs.sum()

    return main_probs, sec_probs

# =============================================================================
# Constraints
# =============================================================================

def violates_constraints(main_nums: List[int], cfg: Config, spec: GameSpec) -> bool:
    arr = np.array(main_nums, dtype=int)
    even = np.sum(arr % 2 == 0)
    if even < cfg.min_even or even > cfg.max_even: return True
    odds = len(arr) - even
    if odds < cfg.min_odd or odds > cfg.max_odd: return True
    lows = np.sum(arr <= spec.main_max // 2)
    if lows < cfg.min_low or lows > cfg.max_low: return True
    s = int(arr.sum())
    if s < cfg.sum_min or s > cfg.sum_max: return True
    consec = 1; best = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            consec += 1; best = max(best, consec)
        else:
            consec = 1
    if best > cfg.max_consecutive: return True
    last = arr % 10
    if np.max(np.bincount(last, minlength=10)) > cfg.max_same_lastdigit: return True
    return False

# =============================================================================
# Sampler (Gumbel Top-k + Wheels)
# =============================================================================

def _gumbel_topk_sample(probs: np.ndarray, k: int, size: int, rng) -> np.ndarray:
    G = rng.gumbel(size=(size, probs.shape[0]))
    scores = np.log(np.clip(probs, 1e-12, 1.0))[None, :] + G
    idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    row = np.arange(idx.shape[0])[:, None]
    order = np.argsort(scores[row, idx], axis=1)[:, ::-1]
    topk = idx[row, order] + 1
    return np.sort(topk, axis=1)

def generate_wheels(key_numbers: List[int], pick_size: int, cfg: Config, spec: GameSpec) -> List[List[int]]:
    wheels = list(itertools.combinations(key_numbers, pick_size))
    return [sorted(w) for w in wheels if not violates_constraints(w, cfg, spec)]

def generate_candidates(df: pd.DataFrame, game: str, cfg: Config, progress=gr.Progress()) -> Tuple[List[Tuple[List[int], object, float]], str]:
    spec = GAMES[game]
    mpb, spb = build_probs(df, game, cfg)
    rng = _rng(cfg.seed)
    candidates = []
    warning = ""

    if cfg.use_wheels:
        try:
            key_numbers = [int(x) for x in cfg.wheel_keys.split(",") if x.strip()]
            if len(key_numbers) >= spec.main_pick:
                wheel_cands = generate_wheels(key_numbers, spec.main_pick, cfg, spec)
                candidates.extend([(m, None, 0.0) for m in wheel_cands])
                warning += f"Generated {len(wheel_cands)} wheel candidates from keys {key_numbers}. "
        except:
            warning += "Invalid wheel keys; skipping wheel generation. "

    size = max(1, int(cfg.iterations - len(candidates)))
    mains = _gumbel_topk_sample(mpb, spec.main_pick, size, rng)
    sec_draw = [None] * size
    if spec.sec_pick == 1:
        probs = spb if spb is not None else np.ones(spec.sec_max)/spec.sec_max
        sec_draw = rng.choice(np.arange(1, spec.sec_max+1), size=size, p=probs)
    elif spec.sec_pick == 2:
        probs = spb if spb is not None else np.ones(spec.sec_max)/spec.sec_max
        Gm = rng.gumbel(size=(size, probs.shape[0]))
        scores = np.log(np.clip(probs, 1e-12, 1.0))[None, :] + Gm
        idx = np.argpartition(scores, -2, axis=1)[:, -2:]
        row = np.arange(idx.shape[0])[:, None]
        order = np.argsort(scores[row, idx], axis=1)[:, ::-1]
        sec_draw = [tuple(sorted((idx[row, order] + 1)[i].tolist())) for i in range(size)]

    lm = np.log(np.clip(mpb, 1e-12, 1.0))
    ls = None if spb is None else np.log(np.clip(spb, 1e-12, 1.0))
    scored = []
    for i in progress.tqdm(range(size), desc="Sampling candidates"):
        m = mains[i].tolist()
        if violates_constraints(m, cfg, spec): continue
        if spec.sec_pick == 0:
            sc = float(np.sum(lm[np.array(m)-1]))
            scored.append((m, None, sc))
        elif spec.sec_pick == 1:
            s = int(sec_draw[i])
            sc = float(np.sum(lm[np.array(m)-1]) + ls[s-1])
            scored.append((m, s, sc))
        else:
            s = sec_draw[i]
            sc = float(np.sum(lm[np.array(m)-1]) + np.sum(ls[np.array(s)-1]))
            scored.append((m, s, sc))

    candidates.extend(scored)
    seen = set(); out = []
    for m, s, sc in sorted(candidates, key=lambda x: x[2], reverse=True):
        key = (tuple(m), s if not isinstance(s, (list, tuple)) else tuple(s))
        if key in seen: continue
        seen.add(key); out.append((m, s, sc))
        if len(out) >= cfg.topk: break
    
    if len(out) < cfg.topk // 2:
        warning += f"Warning: Only {len(out)} candidates generated (target {cfg.topk}). Consider relaxing constraints."
    
    return out, warning

# =============================================================================
# Portfolio (DPP / Greedy + Monte Carlo Risk)
# =============================================================================

def _feat_vec(mains: List[int], sec, spec: GameSpec) -> np.ndarray:
    v = np.zeros(spec.main_max + max(0, spec.sec_max), dtype=float)
    for n in mains: v[n-1] = 1.0
    if spec.sec_pick == 1 and sec is not None:
        v[spec.main_max + (sec-1)] = 0.5
    elif spec.sec_pick == 2 and sec is not None:
        for e in sec: v[spec.main_max + (e-1)] = 0.35
    return v

def dpp_select(cands: List[Tuple[List[int], object, float]], game: str, cfg: Config) -> List[Tuple[List[int], object, float]]:
    spec = GAMES[game]
    k = min(cfg.portfolio_size, len(cands))
    if k <= 0: return []
    feats = np.stack([_feat_vec(m, s, spec) for (m,s,sc) in cands])
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.maximum(norms, 1e-9)
    K = feats @ feats.T
    scores = np.array([sc for (_,_,sc) in cands])
    scores = scores - scores.min() + 1e-3
    q = np.sqrt(scores)
    L = (q[:,None]) * K * (q[None,:])

    coverage = np.sum([np.sum(_feat_vec(m, s, spec)) for m, s, sc in cands])
    L = L * (1.0 + 0.1 * coverage)

    selected = []
    diag = np.diag(L).copy()
    eps = 1e-12
    for _ in range(k):
        i = int(np.argmax(diag))
        if diag[i] <= eps: break
        selected.append(i)
        li = L[:, i][:, None]
        denom = max(eps, L[i,i])
        L = L - (li @ li.T) / denom
        np.fill_diagonal(L, np.maximum(np.diag(L), 0.0))
        diag = np.diag(L)
    return [cands[i] for i in selected]

def monte_carlo_risk(portfolio: List[Tuple[List[int], object, float]], main_probs: np.ndarray, sec_probs: Optional[np.ndarray], spec: GameSpec, cfg: Config) -> Tuple[float, float]:
    def sim_worker(port, mpb, spb, spec, seed, n_sims):
        rng = np.random.default_rng(seed)
        hits = []
        for _ in range(n_sims):
            sim_main = rng.choice(np.arange(1, spec.main_max+1), size=spec.main_pick, p=mpb, replace=False)
            sim_main.sort()
            best_hit = max(len(set(sim_main) & set(m)) for m, _, _ in port)
            hits.append(best_hit)
        return hits

    nproc = 1 if _in_colab() else max(1, mp.cpu_count())
    n_sims_per_proc = max(1, cfg.monte_sims // nproc)
    if nproc == 1:
        hits = sim_worker(portfolio, main_probs, sec_probs, spec, cfg.seed, cfg.monte_sims)
    else:
        pool = mp.Pool(nproc)
        results = [pool.apply_async(sim_worker, args=(portfolio, main_probs, sec_probs, spec, cfg.seed+i, n_sims_per_proc))
                   for i in range(nproc)]
        hits = []
        for r in results:
            hits.extend(r.get())
        pool.close(); pool.join()
    mean_hit = float(np.mean(hits)) if hits else 0.0
    risk_score = float(np.std(hits)) if hits else 0.0
    return mean_hit, risk_score

# =============================================================================
# Walk-forward CV & Self-learning replay
# =============================================================================

def evaluate_cv(df: pd.DataFrame, game: str, cfg: Config, folds: int = 10) -> pd.DataFrame:
    if len(df) < (folds + 10):
        return pd.DataFrame()
    n = len(df)
    idxs = np.linspace(0, n-1, num=folds+1, dtype=int)
    rows = []
    for f in range(folds):
        train_end = idxs[f+1]
        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end: min(train_end+1, n)].copy()
        if len(test)==0 or len(train) < 30: continue
        cands, _ = generate_candidates(train, game, cfg)
        port = dpp_select(cands, game, cfg) if cfg.optimizer=="DPP" else cands[:cfg.portfolio_size]
        t = test.iloc[0]
        spec = GAMES[game]
        test_main = set([int(t[c]) for c in ["n1","n2","n3","n4","n5"]] + (["n6"] if spec.name=="LOTTO" else []))
        bonus = None
        if spec.sec_pick == 1 and "joker" in test.columns:
            bonus = int(t["joker"])
        elif spec.sec_pick == 2 and {"e1","e2"}.issubset(test.columns):
            bonus = (int(t["e1"]), int(t["e2"]))
        best_hit = 0; bonus_hit = 0
        for m, s, _ in port:
            h = len(test_main.intersection(set(m)))
            best_hit = max(best_hit, h)
            if spec.sec_pick == 1 and s is not None and bonus is not None:
                if int(s) == int(bonus): bonus_hit = 1
            elif spec.sec_pick == 2 and s is not None and isinstance(bonus, tuple):
                if len(set(s).intersection(set(bonus))) >= 1: bonus_hit = 1
        rows.append({"fold": f+1, "train_size": len(train), "hit@max": best_hit, "bonus_hit": bonus_hit})
    return pd.DataFrame(rows)

def self_learning_replay(df: pd.DataFrame, game: str, cfg: Config, rounds: int = 2) -> Dict[str, float]:
    beta, gamma = cfg.luck_beta, cfg.unluck_gamma
    spec = GAMES[game]
    score = 0.0
    if len(df) < 100:
        return {"beta": beta, "gamma": gamma, "score": score}
    step_beta = 0.01; step_gamma = 0.005
    for _ in range(rounds):
        hits = []
        idxs = range(80, len(df)-1)
        for j in idxs:
            cfg.luck_beta, cfg.unluck_gamma = beta, gamma
            train = df.iloc[:j].copy()
            test = df.iloc[j:j+1].copy()
            cands, _ = generate_candidates(train, game, cfg)
            port = dpp_select(cands, game, cfg) if cfg.optimizer=="DPP" else cands[:cfg.portfolio_size]
            t = test.iloc[0]
            tmain = set([int(t[c]) for c in ["n1","n2","n3","n4","n5"]] + (["n6"] if spec.name=="LOTTO" else []))
            bh = 0
            for m, s, _ in port:
                bh = max(bh, len(tmain.intersection(set(m))))
            hits.append(bh)
        avg_hit = float(np.mean(hits)) if hits else 0.0
        baseline = 1.2 if spec.name!="LOTTO" else 1.4
        if avg_hit < baseline:
            beta = min(0.5, beta + step_beta)
            gamma = max(0.0, gamma - step_gamma)
            score -= 1.0
        else:
            score += 1.0
    cfg.luck_beta, cfg.unluck_gamma = beta, gamma
    return {"beta": beta, "gamma": gamma, "score": score}

# =============================================================================
# Expected Value (EV)
# =============================================================================

def expected_value_for_ticket(game: str, tiers: List[Dict], ticket_price: float,
                              main_nums: List[int], sec_nums, main_probs: np.ndarray,
                              sec_probs: Optional[np.ndarray]) -> float:
    spec = GAMES[game]
    ev = 0.0
    main_indices = [n-1 for n in main_nums]
    main_prob = np.prod([main_probs[i] for i in main_indices]) if main_probs is not None else 1.0 / math.comb(spec.main_max, spec.main_pick)
    for tier in tiers:
        km = int(tier.get("main", 0))
        ks = int(tier.get("sec", 0))
        prize = float(tier.get("prize", 0.0))
        pm = main_prob if km == spec.main_pick else 0.0
        ps = 1.0
        if spec.sec_pick == 0:
            ps = 1.0 if ks == 0 else 0.0
        elif spec.sec_pick == 1 and sec_nums is not None:
            if sec_probs is not None:
                ps = sec_probs[sec_nums-1] if ks == 1 else (1.0 - sec_probs[sec_nums-1]) if ks == 0 else 0.0
            else:
                ps = 1.0 / spec.sec_max if ks == 1 else (1.0 - 1.0 / spec.sec_max) if ks == 0 else 0.0
        elif spec.sec_pick == 2 and sec_nums is not None:
            if sec_probs is not None:
                sec_indices = [n-1 for n in sec_nums]
                ps = np.prod([sec_probs[i] for i in sec_indices]) if ks == 2 else 0.0
            else:
                ps = math.comb(2, ks) * math.comb(spec.sec_max - 2, 2 - ks) / math.comb(spec.sec_max, 2)
        ev += prize * pm * ps
    return ev - float(ticket_price)

def apply_ev_rerank(game: str, cfg: Config, portfolio: List[Tuple[List[int], object, float]],
                    main_probs: np.ndarray, sec_probs: Optional[np.ndarray]) -> List[Tuple[List[int], object, float, float]]:
    if not cfg.enable_ev:
        return [(m, s, sc, 0.0) for (m, s, sc) in portfolio]
    try:
        tiers = json.loads(cfg.ev_tiers_json or "[]")
        if not isinstance(tiers, list): tiers = []
    except:
        tiers = []
    if not tiers:
        return [(m, s, sc, 0.0) for (m, s, sc) in portfolio]
    out = []
    for (m, s, sc) in portfolio:
        nev = expected_value_for_ticket(game, tiers, cfg.ev_ticket_price, m, s, main_probs, sec_probs)
        out.append((m, s, sc, float(nev)))
    out.sort(key=lambda x: (cfg.ev_weight * x[3] + x[2]), reverse=True)
    return out

# =============================================================================
# Plots & Exports
# =============================================================================

def plot_frequency(df: pd.DataFrame, game: str):
    spec = GAMES[game]
    counts = np.zeros(spec.main_max, dtype=int)
    itcols = ["n1","n2","n3","n4","n5"] + (["n6"] if spec.name=="LOTTO" else [])
    for _, r in df.iterrows():
        for c in itcols:
            counts[int(r[c])-1] += 1
    fig = plt.figure(figsize=(10,3))
    plt.bar(np.arange(1, spec.main_max+1), counts, color='skyblue', edgecolor='navy')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.title(f"{game} — Frequency of Main Numbers", fontsize=12, pad=10)
    plt.xlabel("Number"); plt.ylabel("Count"); plt.tight_layout()
    return fig

def plot_recency(df: pd.DataFrame, game: str):
    spec = GAMES[game]
    last = np.full(spec.main_max, -1, dtype=int)
    for i, (_, r) in enumerate(df.iterrows()):
        for c in ["n1","n2","n3","n4","n5"]:
            last[int(r[c])-1] = i
        if spec.name == "LOTTO":
            last[int(r["n6"])-1] = i
    rec = np.where(last>=0, last, 0)
    if rec.max()>0: rec = rec / rec.max()
    fig = plt.figure(figsize=(10,3))
    plt.bar(np.arange(1, spec.main_max+1), rec, color='lightcoral', edgecolor='darkred')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.title(f"{game} — Recency (Normalized Last Occurrence)", fontsize=12, pad=10)
    plt.xlabel("Number"); plt.ylabel("Recency"); plt.tight_layout()
    return fig

def plot_last_digit(df: pd.DataFrame, game: str):
    spec = GAMES[game]
    digs = np.zeros(10, dtype=int)
    itcols = ["n1","n2","n3","n4","n5"] + (["n6"] if spec.name=="LOTTO" else [])
    for _, r in df.iterrows():
        for c in itcols:
            digs[int(r[c]) % 10] += 1
    fig = plt.figure(figsize=(6,3))
    plt.bar(np.arange(0,10), digs, color='lightgreen', edgecolor='darkgreen')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.title(f"{game} — Last-Digit Distribution", fontsize=12, pad=10)
    plt.xlabel("Last Digit"); plt.ylabel("Count"); plt.tight_layout()
    return fig

def plot_pairs_heatmap(df: pd.DataFrame, game: str):
    spec = GAMES[game]
    M = spec.main_max
    mat = np.zeros((M,M), dtype=int)
    itcols = ["n1","n2","n3","n4","n5"] + (["n6"] if spec.name=="LOTTO" else [])
    for _, r in df.iterrows():
        vals = sorted([int(r[c]) for c in itcols])
        for a, b in itertools.combinations(vals, 2):
            mat[a-1, b-1] += 1
            mat[b-1, a-1] += 1
    thr = np.quantile(mat[mat>0], 0.80) if (mat>0).sum()>0 else 0
    mask = (mat >= thr).astype(int) * mat
    fig = plt.figure(figsize=(6,5))
    plt.imshow(mask, aspect='auto', cmap='viridis')
    plt.colorbar(label="Co-occur Count")
    plt.grid(False)
    plt.title(f"{game} — Significant Pairs (>=80th Percentile)", fontsize=12, pad=10)
    plt.xlabel("Number"); plt.ylabel("Number"); plt.tight_layout()
    return fig

def plot_odd_even(df: pd.DataFrame, game: str):
    spec = GAMES[game]
    itcols = ["n1","n2","n3","n4","n5"] + (["n6"] if spec.name=="LOTTO" else [])
    odds = df[itcols].apply(lambda row: int(np.sum(np.array(row) % 2 == 1)), axis=1)
    fig = plt.figure(figsize=(6,3))
    plt.hist(odds, bins=range(0, spec.main_pick+2), color='purple', edgecolor='black', align='left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.title(f"{game} — Odd/Even Distribution", fontsize=12, pad=10)
    plt.xlabel("Number of Odd Numbers"); plt.ylabel("Frequency"); plt.tight_layout()
    return fig

def export_six_to_csv(df: pd.DataFrame, game: str, prefix: str) -> str:
    ts = _now_ts()
    export_dir = _ensure_game_export_dir(game)
    path = os.path.join(export_dir, f"{prefix}_{ts}.csv")
    df.to_csv(path, index=False)
    return path

def export_six_to_png(df: pd.DataFrame, game: str, title: str, prefix: str) -> str:
    ts = _now_ts()
    export_dir = _ensure_game_export_dir(game)
    path = os.path.join(export_dir, f"{prefix}_{ts}.png")
    plt.figure(figsize=(8, 2 + 0.4*len(df)))
    plt.axis('off')
    headers = list(df.columns)
    cells = df.astype(str).values.tolist()
    table = plt.table(cellText=cells, colLabels=headers, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.3)
    plt.title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    return path

# =============================================================================
# Gradio UI
# =============================================================================

def make_game_tab(game: str):
    spec = GAMES[game]
    state_hist = gr.State(pd.DataFrame())
    state_port = gr.State(pd.DataFrame())
    state_log = gr.State("")
    state_beta = gr.State(0.10)
    state_gamma = gr.State(0.05)
    state_plot_cache = gr.State({})  # plot cache per tab

    with gr.Tab(game):
        gr.Markdown(f"### {game} — Load history, train, and get 6 recommended columns")
        gr.Markdown("**Disclaimer**: Lotteries are random; predictions are probabilistic and not guaranteed. Play responsibly.")

        with gr.Row():
            use_online = gr.Checkbox(value=False, label="Fetch online history (optional)")
            btn_reload = gr.Button("Reload history", variant="secondary")
            btn_preset = gr.Button("Preset: Max Run", variant="secondary")

        hist_log = gr.Markdown(label="History Load Log")
        hist_preview = gr.Dataframe(label=f"{game} — History preview (first 200 rows)", interactive=False, height=220)

        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                iterations = gr.Slider(1000, 500000, value=50000, step=1000, label="Iterations (samples)")
                topk = gr.Slider(10, 1000, value=200, step=10, label="Top-K candidates")
                seed = gr.Number(value=42, label="Seed")
            with gr.Row():
                use_bma = gr.Checkbox(value=True, label="Use BMA")
                use_ewma = gr.Checkbox(value=True, label="Use EWMA")
                use_ml = gr.Checkbox(value=False, label="Use ML hook (experimental, includes Prophet)")
                half_life = gr.Slider(20, 300, value=120, step=5, label="EWMA Half-life (draws)")
            with gr.Row():
                bma_w_freq = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="BMA weight — freq")
                bma_w_rec  = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="BMA weight — recency/EWMA")
                bma_w_ml   = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="BMA weight — ML")
            with gr.Row():
                luck_beta = gr.Slider(0.0, 0.5, value=0.10, step=0.01, label="Luck β (drought boost)")
                unluck_gamma = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="Unluck γ (recent damp)")
            with gr.Row():
                min_even = gr.Slider(0, spec.main_pick, value=2, step=1, label="Min even")
                max_even = gr.Slider(0, spec.main_pick, value=3, step=1, label="Max even")
                min_odd  = gr.Slider(0, spec.main_pick, value=2, step=1, label="Min odd")
                max_odd  = gr.Slider(0, spec.main_pick, value=3, step=1, label="Max odd")
            with gr.Row():
                min_low = gr.Slider(0, spec.main_pick, value=2, step=1, label=f"Min low (1-{spec.main_max//2})")
                max_low = gr.Slider(0, spec.main_pick, value=3, step=1, label="Max low")
                max_same_lastdigit = gr.Slider(1, 5, value=3, step=1, label="Max same last-digit")
            with gr.Row():
                sum_min = gr.Slider(30, 320, value=50 if spec.name!="LOTTO" else 80, step=1, label="Sum min")
                sum_max = gr.Slider(80, 340, value=190 if spec.name!="LOTTO" else 240, step=1, label="Sum max")
                max_consecutive = gr.Slider(1, 6, value=3, step=1, label="Max consecutive")
            with gr.Row():
                optimizer = gr.Dropdown(choices=["DPP","Greedy"], value="DPP", label="Portfolio selector")
                portfolio_size = gr.Slider(1, 12, value=6, step=1, label="Portfolio size (columns)")
                use_wheels = gr.Checkbox(value=False, label="Use lottery wheels")
                wheel_keys = gr.Textbox(value="", label="Wheel key numbers (comma-separated, e.g., 1,2,3,4,5,6)")
            with gr.Row():
                monte_sims = gr.Slider(1000, 50000, value=10000, step=1000, label="Monte Carlo simulations")

        with gr.Accordion("EV re-rank (optional, manual tiers, cost-aware)", open=False):
            enable_ev = gr.Checkbox(value=False, label="Enable EV re-rank")
            ev_weight = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="EV weight vs model score")
            with gr.Row():
                ev_ticket_price = gr.Number(value=OPAP_TICKET_PRICE_DEFAULTS.get(game, 2.0), label="Ticket price (EUR)")
                btn_set_opap_price = gr.Button("Use OPAP default for this game", variant="secondary")
            ev_tiers_json = gr.Code(
                value="[]",
                language="json",
                label="Prize tiers JSON (list of {\"main\":k,\"sec\":s,\"prize\":amount})"
            )

        with gr.Row():
            btn_train = gr.Button("Train (walk-forward CV)", variant="secondary")
            folds = gr.Slider(4, 16, value=10, step=1, label="CV folds")
            btn_learn = gr.Button("Self-learning replay (light)", variant="secondary")

        # Debounce-style feedback label (no heavy compute on drag)
        pending_lbl = gr.Markdown("", visible=False)

        cv_out = gr.Dataframe(label="Walk-forward CV results", interactive=False, height=180)
        learn_out = gr.Markdown("")

        with gr.Row():
            btn_predict = gr.Button("Predict Portfolio", variant="primary")
            btn_export_csv = gr.Button("Export 6 columns → CSV", variant="secondary")
            btn_export_png = gr.Button("Export 6 columns → PNG", variant="secondary")

        warning_out = gr.Markdown("")
        risk_out = gr.Markdown("")
        portfolio_df = gr.Dataframe(label="Recommended 6 columns", interactive=False, height=200)
        export_links = gr.Markdown("")

        with gr.Row():
            plot1 = gr.Plot(label="Frequency")
            plot2 = gr.Plot(label="Recency")
            plot3 = gr.Plot(label="Last-digit distribution")
        with gr.Row():
            plot4 = gr.Plot(label="Significant pairs heatmap")
            plot5 = gr.Plot(label="Odd/Even distribution")

        # --- inner fns ---
        def _reload(use_online_flag):
            df, log = _load_all_history(game, use_online=bool(use_online_flag))
            return log, df.head(200), df

        def _preset():
            return 200000, 500, 777, True, True, False, 160, 0.5, 0.3, 0.2, 0.12, 0.06

        def _set_opap_price():
            return float(OPAP_TICKET_PRICE_DEFAULTS.get(game, 2.0))

        def _collect_cfg(iterations_v, topk_v, seed_v, use_bma_v, use_ewma_v, use_ml_v,
                         half_life_v, bma_wf, bma_wr, bma_wml, luck_b, unluck_g,
                         min_e, max_e, min_o, max_o, min_l, max_l, max_ld, smin, smax, maxc,
                         opt, pk, use_wheels_v, wheel_keys_v, enable_ev_v, ev_w, ev_price, ev_json,
                         use_online_flag, monte_sims_v):
            cfg = Config(
                iterations=int(iterations_v), topk=int(topk_v), seed=int(seed_v),
                use_bma=bool(use_bma_v), bma_w_freq=float(bma_wf), bma_w_rec=float(bma_wr), bma_w_ml=float(bma_wml),
                use_ewma=bool(use_ewma_v), half_life=int(half_life_v), use_ml=bool(use_ml_v),
                luck_beta=float(luck_b), unluck_gamma=float(unluck_g),
                min_even=int(min_e), max_even=int(max_e), min_odd=int(min_o), max_odd=int(max_o),
                min_low=int(min_l), max_low=int(max_l), max_same_lastdigit=int(max_ld),
                sum_min=int(smin), sum_max=int(smax), max_consecutive=int(maxc),
                optimizer=str(opt), portfolio_size=int(pk),
                use_wheels=bool(use_wheels_v), wheel_keys=str(wheel_keys_v or ""),
                enable_ev=bool(enable_ev_v), ev_weight=float(ev_w), ev_ticket_price=float(ev_price),
                ev_tiers_json=str(ev_json or "[]"),
                use_online=bool(use_online_flag), monte_sims=int(monte_sims_v)
            )
            return cfg

        def _train(df, *cfg_flat, folds=10):
            if df is None or len(df)==0: return pd.DataFrame()
            cfg = _collect_cfg(*cfg_flat)
            res = evaluate_cv(df, game, cfg, folds=int(folds))
            return res

        def _learn(df, *cfg_flat):
            if df is None or len(df)==0:
                return "No history loaded.", state_beta.value, state_gamma.value
            cfg = _collect_cfg(*cfg_flat)
            stats = self_learning_replay(df, game, cfg, rounds=2)
            msg = f"**Self-learning replay done**  \nβ={stats['beta']:.3f}, γ={stats['gamma']:.3f}, score={stats['score']:+.1f}"
            return msg, stats['beta'], stats['gamma']

        def _predict(df, *cfg_flat, progress=gr.Progress()):
            if df is None or len(df)==0:
                return pd.DataFrame(), None, None, None, None, None, "No history loaded.", "", None, state_beta.value, state_gamma.value, state_plot_cache.value
            cfg = _collect_cfg(*cfg_flat)
            cands, warning = generate_candidates(df, game, cfg, progress=progress)
            main_probs, sec_probs = build_probs(df, game, cfg)
            port = dpp_select(cands, game, cfg) if cfg.optimizer=="DPP" else cands[:cfg.portfolio_size]
            mean_hit, risk_score = monte_carlo_risk(port, main_probs, sec_probs, spec, cfg)
            baseline = 1.2 if spec.name!="LOTTO" else 1.4
            cfg.luck_beta += 0.01 if mean_hit < baseline else -0.01
            cfg.luck_beta = max(0.0, min(0.5, cfg.luck_beta))
            cfg.unluck_gamma += 0.005 if mean_hit < baseline else -0.005
            cfg.unluck_gamma = max(0.0, min(0.5, cfg.unluck_gamma))
            port_ev = apply_ev_rerank(game, cfg, port, main_probs, sec_probs)
            port_ev = sorted(port_ev, key=lambda x: x[2] - 0.5 * risk_score, reverse=True)
            rows = []
            headers = [f"n{i+1}" for i in range(spec.main_pick if spec.name!="LOTTO" else 6)]
            if spec.sec_pick==1: headers += ["bonus"]
            elif spec.sec_pick==2: headers += ["e1","e2"]
            for m,s,sc,nev in port_ev[:6]:
                row = list(m)
                if spec.sec_pick==1: row += [s]
                elif spec.sec_pick==2: row += list(s)
                rows.append(row)
            out = pd.DataFrame(rows, columns=headers)

            # --- Plot caching ---
            cache_key = _quick_df_sig(df, game)
            cache: dict = state_plot_cache.value or {}
            cached = cache.get(cache_key, {})
            if all(k in cached for k in ("fig1","fig2","fig3","fig4","fig5")):
                fig1, fig2, fig3, fig4, fig5 = (
                    cached["fig1"], cached["fig2"], cached["fig3"], cached["fig4"], cached["fig5"]
                )
            else:
                fig1 = plot_frequency(df, game)
                fig2 = plot_recency(df, game)
                fig3 = plot_last_digit(df, game)
                fig4 = plot_pairs_heatmap(df, game)
                fig5 = plot_odd_even(df, game)
                cached = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}
                cache[cache_key] = cached
                state_plot_cache.value = cache

            risk_msg = f"Portfolio Risk: Mean Hits = {mean_hit:.2f}, Std = {risk_score:.2f} (lower is safer)\\nUpdated β={cfg.luck_beta:.3f}, γ={cfg.unluck_gamma:.3f}"
            return out, out, fig1, fig2, fig3, fig5, warning, risk_msg, fig4, cfg.luck_beta, cfg.unluck_gamma, state_plot_cache.value

        def _export(df, game_label=game):
            if df is None or len(df)==0:
                return f"{game_label}: Nothing to export."
            p_csv = export_six_to_csv(df, game_label, prefix=f"{game_label.lower()}_six")
            p_png = export_six_to_png(df, game_label, title=f"{game_label} — 6 Recommended Columns", prefix=f"{game_label.lower()}_six")
            return f"**{game_label}** — **CSV:** [{os.path.basename(p_csv)}](/file={p_csv})  \\n**PNG:** [{os.path.basename(p_png)}](/file={p_png})"

        # --- debounce-style feedback (lightweight) ---
        def _mark_pending(_=None):
            return "**Settings changed** (will apply on next Predict/Train).", True
        def _clear_pending():
            return "", False

        for sl in (iterations, topk, half_life, monte_sims):
            sl.input(_mark_pending, outputs=[pending_lbl, gr.State(True)])
            sl.release(_clear_pending, outputs=[pending_lbl, gr.State(False)])

        # --- wiring ---
        btn_reload.click(_reload, inputs=[use_online], outputs=[hist_log, hist_preview, state_hist])
        btn_preset.click(_preset, outputs=[
            iterations, topk, seed, use_bma, use_ewma, use_ml, half_life, bma_w_freq, bma_w_rec, bma_w_ml, luck_beta, unluck_gamma
        ])
        btn_set_opap_price.click(_set_opap_price, outputs=[ev_ticket_price])

        btn_train.click(
            _train,
            inputs=[state_hist, iterations, topk, seed, use_bma, use_ewma, use_ml, half_life, bma_w_freq, bma_w_rec, bma_w_ml,
                    luck_beta, unluck_gamma, min_even, max_even, min_odd, max_odd, min_low, max_low, max_same_lastdigit,
                    sum_min, sum_max, max_consecutive, optimizer, portfolio_size, use_wheels, wheel_keys,
                    enable_ev, ev_weight, ev_ticket_price, ev_tiers_json, use_online, monte_sims, folds],
            outputs=[cv_out]
        )

        btn_learn.click(
            _learn,
            inputs=[state_hist, iterations, topk, seed, use_bma, use_ewma, use_ml, half_life, bma_w_freq, bma_w_rec, bma_w_ml,
                    luck_beta, unluck_gamma, min_even, max_even, min_odd, max_odd, min_low, max_low, max_same_lastdigit,
                    sum_min, sum_max, max_consecutive, optimizer, portfolio_size, use_wheels, wheel_keys,
                    enable_ev, ev_weight, ev_ticket_price, ev_tiers_json, use_online, monte_sims],
            outputs=[learn_out, state_beta, state_gamma]
        )

        btn_predict.click(
            _predict,
            inputs=[state_hist, iterations, topk, seed, use_bma, use_ewma, use_ml, half_life, bma_w_freq, bma_w_rec, bma_w_ml,
                    luck_beta, unluck_gamma, min_even, max_even, min_odd, max_odd, min_low, max_low, max_same_lastdigit,
                    sum_min, sum_max, max_consecutive, optimizer, portfolio_size, use_wheels, wheel_keys,
                    enable_ev, ev_weight, ev_ticket_price, ev_tiers_json, use_online, monte_sims],
            outputs=[portfolio_df, state_port, plot1, plot2, plot3, plot5, warning_out, risk_out, plot4, state_beta, state_gamma, state_plot_cache]
        )

        btn_export_csv.click(_export, inputs=[state_port], outputs=[export_links])
        btn_export_png.click(_export, inputs=[state_port], outputs=[export_links])

    return state_hist, state_port, state_log, state_beta, state_gamma

def build_history_tab():
    with gr.Tab("History"):
        gr.Markdown(textwrap.dedent("""
        ### History folders (place files before reloading each game)
        - `/content/data/history/tzoker/` — CSV/XLSX cols: **n1..n5, joker** (optional `date`)
        - `/content/data/history/lotto/` — CSV/XLSX cols: **n1..n6** (optional `date`)
        - `/content/data/history/eurojackpot/` — CSV/XLSX cols: **n1..n5, e1, e2** (optional `date`)

        Multiple files auto-merge, rows are validated and normalized. Case-insensitive headers.
        Include a `date` column (e.g., YYYY-MM-DD) for best results with Prophet (time-series model in ML ensemble, used when `use_ml=True` and ≥120 draws with valid dates are available).
        Use optional online fetch for a quick refresh; local files are recommended for reliability.

        **Colab Instructions**:
        1. Run the setup cell below to install dependencies.
        2. Upload history files to `/content/data/history/<game>/` using the Colab file uploader.
        3. Alternatively, enable **Fetch online** in each game tab.

        ```bash
        !pip -q install numpy pandas matplotlib gradio requests lxml
        !pip -q install lightgbm xgboost prophet scikit-learn
        !mkdir -p /content/data/history/tzoker /content/data/history/lotto /content/data/history/eurojackpot
        !mkdir -p /content/exports/tzoker /content/exports/lotto /content/exports/eurojackpot
        ```
        """))

def build_export_tab(states: Dict[str, gr.State]):
    with gr.Tab("Export"):
        gr.Markdown("Export last 6 columns for each game to /content/exports/<game>/.")

        with gr.Row():
            btn_tz = gr.Button("Export TZOKER")
            btn_lo = gr.Button("Export LOTTO")
            btn_ej = gr.Button("Export EUROJACKPOT")
        md = gr.Markdown("")

        def _exp(df, label):
            if df is None or len(df) == 0:
                return f"{label}: Nothing to export."
            p_csv = export_six_to_csv(df, label, prefix=f"{label.lower()}_six")
            p_png = export_six_to_png(df, label, title=f"{label} — 6 Recommended Columns", prefix=f"{label.lower()}_six")
            return f"**{label}** — **CSV:** [{os.path.basename(p_csv)}](/file={p_csv})  \\n**PNG:** [{os.path.basename(p_png)}](/file={p_png})"

        btn_tz.click(_exp, inputs=[states["TZOKER_port"], gr.State("TZOKER")], outputs=[md])
        btn_lo.click(_exp, inputs=[states["LOTTO_port"], gr.State("LOTTO")], outputs=[md])
        btn_ej.click(_exp, inputs=[states["EUROJACKPOT_port"], gr.State("EUROJACKPOT")], outputs=[md])

def build_settings_tab():
    with gr.Tab("Settings"):
        gr.Markdown(textwrap.dedent("""
        **Global notes**

        - Offline by default; enable online fetching for fresh data.
        - For heavy runs, use Preset: Max Run in each game tab.
        - ML hook is experimental and off by default; it requires ample data (≥120 draws) and libraries (lightgbm, scikit-learn, xgboost, prophet).
        - Prophet (time-series model) enhances ML predictions with temporal trends when `use_ml=True` and ≥120 draws with valid date columns are available.
        - Adaptive constraints auto-set from history; override if you prefer stricter/looser filters.
        - Ensure history files are clean (deduped, correct ranges).

        **Tips**

        - Increase iterations and top-k for more stable candidate pools.
        - Use DPP (with coverage boost) to diversify better than greedy.
        - Wheels can enforce coverage of your key numbers.
        - Download CSV/PNG outputs from `/content/exports/<game>/` in Colab, as files are temporary.
        """))

if __name__ == "__main__":
    with gr.Blocks(title="Ultra Lottery Helper") as demo:
        states: Dict[str, gr.State] = {}
        for game in ["TZOKER", "LOTTO", "EUROJACKPOT"]:
            s_hist, s_port, s_log, s_beta, s_gamma = make_game_tab(game)
            states[game] = s_hist                 # history state (for reference)
            states[f"{game}_port"] = s_port       # used by Export tab
            states[f"{game}_log"] = s_log
            states[f"{game}_beta"] = s_beta
            states[f"{game}_gamma"] = s_gamma

        build_history_tab()
        build_export_tab(states)
        build_settings_tab()
    demo.launch(share=_in_colab(), quiet=True)
