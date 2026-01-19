from __future__ import annotations



def _last_report_path() -> Path:
    return LEARN_DIR / "last_report.json"

def get_status_summary() -> dict:
    """
    Returns a compact status dict with last learn metrics (if any) and current state.
    """
    state = _load_state()
    report = {}
    p = _last_report_path()
    if p.exists():
        try:
            report = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            report = {}
    return {"state": state, "report": report}


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ulh_learning.py — Post-draw learning & continuous improvement utilities for Oracle Lottery Predictor.

What it does
------------
• Records portfolios you generated (predictions) into a local SQLite DB.
• Records actual draw outcomes.
• Compares predictions vs outcomes and computes hit metrics.
• Adjusts model state (luck/unluck, EWMA half-life, ensemble weights) and persists them.
• Exposes a simple API + a CLI (see ulh_learn_cli.py).

Design notes
------------
- Non-intrusive: does not change core APIs of ultra_lottery_helper.py.
- Persists everything under ./data/learning/ (SQLite + JSON).
- You can import and call these functions from the desktop app if desired.
"""
import os, json, sqlite3, datetime as dt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from src.ultra_lottery_helper import (
    Config, GAMES, generate_candidates, apply_ev_rerank,
    expected_value_for_ticket, self_learning_replay, _load_all_history
)

ROOT = Path(__file__).resolve().parents[1]
LEARN_DIR = ROOT / "data" / "learning"
LEARN_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = LEARN_DIR / "ulh_learning.sqlite"
STATE_JSON = LEARN_DIR / "model_state.json"

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game TEXT NOT NULL,
            ts_utc TEXT NOT NULL,
            tag TEXT,
            combo TEXT NOT NULL  -- JSON array of ints (main numbers), optional sec in object
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game TEXT NOT NULL,
            ts_draw TEXT NOT NULL,  -- date or datetime as ISO
            outcome TEXT NOT NULL   -- JSON: {"main":[...], "sec":[...]}
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game TEXT NOT NULL,
            ts_draw TEXT NOT NULL,
            k INTEGER NOT NULL,
            hit_main INTEGER NOT NULL,
            hit_sec INTEGER NOT NULL,
            combo TEXT NOT NULL      -- JSON of the evaluated combo
        )
    """)
    return conn

def _utcnow():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat()

def _load_state() -> Dict:
    """Load learning state from JSON file."""
    if STATE_JSON.exists():
        try:
            return json.loads(STATE_JSON.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            # Log and return defaults if state file is corrupted
            import logging
            logging.warning(f"Failed to load state from {STATE_JSON}: {e}")
    # defaults
    return {
        "luck_beta": 0.10,
        "unluck_gamma": 0.05,
        "half_life": 180,
        "ensemble": {"ewma": 0.30, "recency": 0.20, "ml": 0.50}
    }

def _save_state(state: Dict):
    """Save learning state to JSON file."""
    STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")

# ----------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------

def record_portfolio(game: str, portfolio: List[List[int]], tag: Optional[str] = None) -> int:
    """
    Save a generated portfolio (list of combinations) for a game.
    Returns the number of saved rows.
    
    Raises:
        ValueError: If game is unknown
    """
    if game not in GAMES:
        raise ValueError(f"Unknown game: {game}")
    ts = _utcnow()
    rows = [(game, ts, tag, json.dumps(c)) for c in portfolio]
    with _connect() as cx:
        cx.executemany("INSERT INTO predictions (game, ts_utc, tag, combo) VALUES (?,?,?,?)", rows)
        cx.commit()
    return len(rows)

def record_outcome(game: str, main: List[int], sec: Optional[List[int]] = None, ts_draw: Optional[str] = None) -> int:
    """
    Record an official draw outcome.
    
    Args:
        game: Game identifier
        main: Main numbers drawn
        sec: Secondary numbers (optional)
        ts_draw: ISO string (date or datetime); default = today (UTC date)
        
    Returns:
        Number of rows inserted (always 1)
        
    Raises:
        ValueError: If game is unknown
    """
    if game not in GAMES:
        raise ValueError(f"Unknown game: {game}")
    if ts_draw is None:
        ts_draw = dt.date.today().isoformat()
    payload = {"main": sorted([int(x) for x in main])}
    if sec:
        payload["sec"] = sorted([int(x) for x in sec])
    with _connect() as cx:
        cx.execute("INSERT INTO outcomes (game, ts_draw, outcome) VALUES (?,?,?)",
                   (game, ts_draw, json.dumps(payload)))
        cx.commit()
    return 1

def evaluate_latest(game: str, k_limit: Optional[int] = None) -> Dict[str, float]:
    """
    Compare the MOST RECENT recorded predictions of `game` vs the MOST RECENT outcome,
    store granular evals, and return summary metrics.
    
    Raises:
        ValueError: If game is unknown
    """
    if game not in GAMES:
        raise ValueError(f"Unknown game: {game}")
    with _connect() as cx:
        # Last outcome
        row = cx.execute("SELECT ts_draw, outcome FROM outcomes WHERE game=? ORDER BY id DESC LIMIT 1", (game,)).fetchone()
        if not row:
            return {"error": "No outcomes recorded yet."}
        ts_draw, outcome_json = row
        outcome = json.loads(outcome_json)
        main_true = set(outcome.get("main", []))
        sec_true = set(outcome.get("sec", []))

        # Last predictions BEFORE or ON that draw time
        rows = cx.execute("SELECT id, combo FROM predictions WHERE game=? ORDER BY id DESC LIMIT 1000", (game,)).fetchall()
        if not rows:
            return {"error": "No predictions recorded yet."}

        # Evaluate each combo
        stats = {"n": 0, "hit_any": 0, "hit_main_total": 0, "hit_sec_total": 0, "hit_top1_main": 0}
        cx.execute("BEGIN")
        for i, (pid, combo_json) in enumerate(rows):
            combo = json.loads(combo_json)
            hit_main = len(main_true.intersection(combo))
            hit_sec = 0  # Many games have at most 1 sec; adapt if needed
            cx.execute("INSERT INTO evals (game, ts_draw, k, hit_main, hit_sec, combo) VALUES (?,?,?,?,?,?)",
                       (game, ts_draw, i+1, hit_main, hit_sec, combo_json))
            stats["n"] += 1
            stats["hit_any"] += 1 if (hit_main + hit_sec) > 0 else 0
            stats["hit_main_total"] += hit_main
            stats["hit_sec_total"] += hit_sec
            if i == 0:
                stats["hit_top1_main"] = hit_main
            if k_limit and i+1 >= k_limit:
                break
        cx.commit()

    # Summary
    n = max(1, stats["n"])
    return {
        "combos_evaluated": n,
        "any_hit_rate": stats["hit_any"] / n,
        "avg_main_hits_per_combo": stats["hit_main_total"] / n,
        "top1_main_hits": stats["hit_top1_main"]
    }

def update_model_state_from_eval(game: str, eval_summary: Dict[str, float]) -> Dict[str, float]:
    """
    Adjust the learning state based on evaluation summary.
    Conservative heuristic updates; safe defaults.
    """
    state = _load_state()
    hr = eval_summary.get("any_hit_rate", 0.0)
    top1 = eval_summary.get("top1_main_hits", 0.0)
    # Simple heuristics: if we rarely hit anything, increase recency + luck penalty.
    if hr < 0.05:
        state["luck_beta"] = min(0.50, state["luck_beta"] + 0.02)
        state["unluck_gamma"] = min(0.30, state["unluck_gamma"] + 0.01)
        state["half_life"] = max(60, int(state["half_life"] * 0.9))  # shorter memory
        # shift ensemble toward recency
        e = state["ensemble"]
        e["recency"] = min(0.6, e["recency"] + 0.05)
        e["ewma"] = max(0.1, e["ewma"] - 0.02)
        e["ml"] = max(0.2, e["ml"] - 0.03)
    elif hr > 0.15 or top1 >= 2:
        # Doing relatively well: lengthen memory and favor ML/EWMA slightly
        state["luck_beta"] = max(0.05, state["luck_beta"] - 0.01)
        state["unluck_gamma"] = max(0.02, state["unluck_gamma"] - 0.005)
        state["half_life"] = min(365, int(state["half_life"] * 1.05))
        e = state["ensemble"]
        e["ml"] = min(0.7, e["ml"] + 0.03)
        e["ewma"] = min(0.5, e["ewma"] + 0.02)
        e["recency"] = max(0.1, e["recency"] - 0.04)
    _save_state(state)
    return state

def apply_state_to_config(cfg: Config) -> Config:
    """
    Mutates cfg in-place to reflect the persisted learning state.
    """
    state = _load_state()
    cfg.luck_beta = float(state.get("luck_beta", cfg.luck_beta))
    cfg.unluck_gamma = float(state.get("unluck_gamma", cfg.unluck_gamma))
    cfg.half_life = int(state.get("half_life", cfg.half_life))
    ens = state.get("ensemble", {})
    cfg.weight_ewma = float(ens.get("ewma", getattr(cfg, "weight_ewma", 0.3)))
    cfg.weight_recency = float(ens.get("recency", getattr(cfg, "weight_recency", 0.2)))
    cfg.weight_ml = float(ens.get("ml", getattr(cfg, "weight_ml", 0.5)))
    return cfg

def learn_after_draw(game: str, k_limit: Optional[int] = None, self_replay_rounds: int = 1) -> Dict[str, float]:
    """
    One-call routine:
      1) evaluate_latest()
      2) update_model_state_from_eval()
      3) optional self_learning_replay() on full history to re-tune luck params
      4) persist new state
      5) return merged metrics & state snapshot
    """
    summary = evaluate_latest(game, k_limit=k_limit)
    if "error" in summary:
        return summary
    state = update_model_state_from_eval(game, summary)

    # Optional deeper replay on full history (walk-forward)
    cfg = Config()
    cfg = apply_state_to_config(cfg)
    hist, _ = _load_all_history(game)
    if not hist.empty:
        _ = self_learning_replay(hist, game, cfg, rounds=self_replay_rounds)

    # Persist final state again (in case replay nudged cfg via luck params)
    state = _load_state()
    state["luck_beta"] = float(cfg.luck_beta)
    state["unluck_gamma"] = float(cfg.unluck_gamma)
    state["half_life"] = int(cfg.half_life)
    _save_state(state)

    merged = dict(summary)
    merged.update({"state": state})
    return merged