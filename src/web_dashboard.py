#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight web dashboard for configuration, status, notifications, and metrics.

The server intentionally uses only the Python standard library to avoid new
dependencies. It exposes a minimal Web UI for:
- Editing scheduler configuration
- Viewing real-time status
- Viewing basic metrics
- Triggering test email/SMS notifications (logged locally)
"""

import argparse
import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:  # Support both package and script execution
    from .utils import get_logger, load_json, save_json
    from .ultra_lottery_helper import LOTTERY_METADATA
except ImportError:  # pragma: no cover - fallback for direct execution
    from utils import get_logger, load_json, save_json  # type: ignore
    from ultra_lottery_helper import LOTTERY_METADATA  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from .lottery_scheduler import LotteryScheduler

LOGGER = get_logger("web_dashboard")
DEFAULT_PORT = 8080
ASSETS_ROOT = Path(__file__).resolve().parent.parent / "assets"
FLAG_EMOJI = {
    "Greece": "🇬🇷",
    "European Union": "🇪🇺",
    "United Kingdom": "🇬🇧",
    "Spain": "🇪🇸",
    "Italy": "🇮🇹",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Austria": "🇦🇹",
    "Switzerland": "🇨🇭",
    "United States": "🇺🇸",
    "Australia": "🇦🇺",
    "Canada": "🇨🇦",
    "Japan": "🇯🇵",
    "South Africa": "🇿🇦",
}


def _default_data_root() -> Path:
    """Return the default data root path."""
    return Path(__file__).resolve().parent.parent / "data"


def _history_root(base: Path) -> Path:
    """Return the history directory, falling back to repository data."""
    if (base / "history").exists():
        return base / "history"
    return Path(__file__).resolve().parent.parent / "data" / "history"


def _load_latest_draw(data_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the newest draw from CSV files, if present."""
    if not data_dir.exists():
        return None

    csv_files: List[Path] = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        csv_files = sorted(data_dir.rglob("*.csv"))
    for csv_file in reversed(csv_files):
        try:
            with csv_file.open("r", encoding="utf-8") as f:
                reader = list(csv.reader(f))
        except (OSError, UnicodeDecodeError):
            continue
        if len(reader) <= 1:
            continue
        # Skip header, find last non-empty row
        for row in reversed(reader[1:]):
            if not any(row):
                continue
            date = row[0]
            numbers = []
            for value in row[1:]:
                try:
                    numbers.append(int(value))
                except (TypeError, ValueError):
                    if value:
                        numbers.append(value)
            return {"date": date, "numbers": numbers, "source": csv_file.name}
    return None


def _prediction_status(history_root: Path, game: str) -> Dict[str, Any]:
    """Summarize prediction tracking status for a game."""
    predictions = load_json(history_root / "predictions.json", default={}, logger=LOGGER)
    results = load_json(history_root / "prediction_results.json", default={}, logger=LOGGER)

    pending = len(predictions.get(game, [])) if isinstance(predictions, dict) else 0
    game_results = results.get(game, []) if isinstance(results, dict) else []
    if isinstance(game_results, dict):
        accuracy = game_results.get("accuracy")
        completed = game_results.get("completed", 0)
    else:
        accuracy = None
        completed = len(game_results) if isinstance(game_results, list) else 0

    return {
        "pending": pending,
        "completed": completed,
        "accuracy": accuracy,
    }


def _build_news_items(lotteries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compose lightweight news feed from lottery snapshots."""
    news: List[Dict[str, Any]] = []
    for entry in lotteries:
        latest = entry.get("latest_draw") or {}
        if not latest:
            continue
        news.append(
            {
                "id": entry["id"],
                "title": f"{entry['display_name']} latest draw",
                "date": latest.get("date"),
                "numbers": latest.get("numbers", []),
                "has_jackpot": entry.get("has_jackpot", False),
                "history_available": entry.get("history_available", False),
            }
        )
    return news


class MetricsCollector:
    """Minimal metrics collector with JSON persistence."""

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else _default_data_root()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.data_root / "metrics.json"
        self._metrics = load_json(
            self.metrics_file, default={"counters": {}, "events": []}, logger=LOGGER
        )

    def record(self, name: str, value: int = 1) -> None:
        """Record a counter-style metric."""
        counters = self._metrics.setdefault("counters", {})
        counters[name] = counters.get(name, 0) + value
        self._metrics["last_updated"] = time.time()
        self.persist()

    def add_event(self, message: str, severity: str = "info") -> None:
        """Append a short event to the rolling log."""
        events = self._metrics.setdefault("events", [])
        events.append(
            {"ts": time.time(), "message": message[:200], "severity": severity[:10]}
        )
        # Keep only the most recent 50 events
        self._metrics["events"] = events[-50:]
        self._metrics["last_updated"] = time.time()
        self.persist()

    def summarize(self) -> Dict[str, Any]:
        """Return a copy of current metrics."""
        return {
            "counters": dict(self._metrics.get("counters", {})),
            "events": list(self._metrics.get("events", [])),
            "last_updated": self._metrics.get("last_updated"),
        }

    def persist(self) -> None:
        """Persist metrics safely to disk."""
        save_json(self.metrics_file, self._metrics, atomic=True, logger=LOGGER)


class NotificationManager:
    """Simple email/SMS notification logger."""

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else _default_data_root()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.log_file = self.data_root / "notifications.json"
        self._log = load_json(self.log_file, default=[], logger=LOGGER)

    def _append(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        entry["ts"] = time.time()
        self._log.append(entry)
        # Keep log small
        self._log = self._log[-100:]
        save_json(self.log_file, self._log, atomic=True, logger=LOGGER)
        return entry

    def send_email(self, to_address: str, subject: str, body: str) -> Dict[str, Any]:
        """Log an email notification request."""
        return self._append(
            {
                "channel": "email",
                "to": to_address,
                "subject": subject[:120],
                "body": body[:500],
                "status": "logged",
            }
        )

    def send_sms(self, phone_number: str, message: str) -> Dict[str, Any]:
        """Log an SMS notification request."""
        return self._append(
            {
                "channel": "sms",
                "to": phone_number,
                "message": message[:280],
                "status": "logged",
            }
        )

    def history(self, limit: int = 10) -> Dict[str, Any]:
        """Return most recent notification requests."""
        recent = self._log[-limit:]
        return {"count": len(self._log), "recent": recent[::-1]}


class DashboardState:
    """Holds shared state for the HTTP handler."""

    def __init__(
        self,
        scheduler: Optional["LotteryScheduler"] = None,
        metrics: Optional[MetricsCollector] = None,
        notifications: Optional[NotificationManager] = None,
        data_root: Optional[Path] = None,
    ):
        self.scheduler = scheduler
        self.metrics = metrics or MetricsCollector(data_root=data_root)
        self.notifications = notifications or NotificationManager(data_root=data_root)
        self.data_root = Path(data_root) if data_root else _default_data_root()

    def status(self) -> Dict[str, Any]:
        """Compose status payload for the dashboard."""
        scheduler_status: Dict[str, Any]
        if self.scheduler:
            try:
                scheduler_status = self.scheduler.get_status()
            except Exception as exc:  # pragma: no cover - defensive
                scheduler_status = {"error": str(exc)}
        else:
            scheduler_status = {"enabled": False, "running": False, "backend": "None"}

        scheduler_status["metrics"] = self.metrics.summarize()
        scheduler_status["notifications"] = self.notifications.history(limit=5)
        return scheduler_status

    def lotteries_overview(self) -> Dict[str, Any]:
        """Return snapshot of all lotteries with metadata and latest draws."""
        history_dir = _history_root(self.data_root)
        lotteries: List[Dict[str, Any]] = []
        for game, meta in LOTTERY_METADATA.items():
            game_dir = history_dir / game.lower()
            latest = _load_latest_draw(game_dir)
            history_available = game_dir.exists() and any(game_dir.rglob("*"))
            prediction = _prediction_status(history_dir, game)
            lotteries.append(
                {
                    "id": game,
                    "display_name": meta.get("display_name", game),
                    "country": meta.get("country", ""),
                    "flag": meta.get("flag"),
                    "flag_emoji": FLAG_EMOJI.get(meta.get("country", ""), "🎯"),
                    "icon": meta.get("icon"),
                    "has_jackpot": bool(meta.get("has_jackpot")),
                    "official_url": meta.get("official_url"),
                    "results_url": meta.get("results_url"),
                    "description": meta.get("description", ""),
                    "latest_draw": latest,
                    "history_available": history_available,
                    "prediction": prediction,
                }
            )

        return {"lotteries": lotteries, "news": _build_news_items(lotteries)}

    def update_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a minimal scheduler configuration update.

        Only a safe subset of keys is allowed to be updated from the Web UI.
        """
        if not self.scheduler:
            return {"error": "Scheduler not available"}

        allowed_keys = {"enabled", "default_interval_hours", "global_settings"}
        for key, value in payload.items():
            if key in allowed_keys:
                self.scheduler.schedule_config[key] = value
        self.scheduler._save_schedule_config()
        return self.scheduler.schedule_config


def _index_html() -> bytes:
    """Serve the minimal dashboard UI."""
    html = """<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Oracle Lottery Dashboard</title>
  <style>
    :root {{
      --bg: #0b1521;
      --panel: #132235;
      --panel-2: #0f1928;
      --border: #1f3b57;
      --accent: #1f8ef1;
      --accent-2: #22c55e;
      --text: #e6edf3;
      --muted: #9fb7d3;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: 'Inter', Arial, sans-serif; background: var(--bg); color: var(--text); }}
    a {{ color: var(--accent); text-decoration: none; }}
    .app {{ display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }}
    .sidebar {{ background: #0d1b2a; border-right: 1px solid var(--border); padding: 16px; display: flex; flex-direction: column; gap: 16px; }}
    .logo {{ font-weight: 700; font-size: 18px; letter-spacing: 0.5px; }}
    .sidebar-section {{ border: 1px solid var(--border); border-radius: 10px; padding: 10px; background: var(--panel); }}
    .sidebar-title {{ font-size: 13px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }}
    .lottery-list {{ display: flex; flex-direction: column; gap: 6px; max-height: calc(100vh - 220px); overflow-y: auto; }}
    .lotto-row {{ display: flex; align-items: center; gap: 10px; padding: 10px; border-radius: 8px; border: 1px solid transparent; background: var(--panel-2); cursor: pointer; text-align: left; }}
    .lotto-row.active {{ border-color: var(--accent); background: rgba(31, 142, 241, 0.12); }}
    .lotto-row:hover {{ border-color: var(--border); }}
    .flag {{ font-size: 18px; }}
    .lotto-names {{ display: flex; flex-direction: column; line-height: 1.2; }}
    .lotto-country {{ color: var(--muted); font-size: 12px; }}
    .sidebar-button {{ width: 100%; padding: 10px; border-radius: 8px; border: 1px solid var(--border); background: var(--panel-2); color: var(--text); cursor: pointer; }}
    .sidebar-button:hover {{ border-color: var(--accent); }}
    .content {{ padding: 20px; display: flex; flex-direction: column; gap: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 12px; }}
    h1, h3 {{ margin: 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .news-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .news-card {{ background: var(--panel-2); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
    .chip {{ background: rgba(31,142,241,0.15); color: var(--text); padding: 4px 8px; border-radius: 20px; font-size: 12px; }}
    .pill {{ padding: 6px 10px; border-radius: 20px; border: 1px solid var(--border); background: var(--panel-2); font-size: 12px; }}
    .title-row {{ display: flex; align-items: center; gap: 10px; }}
    .icon-circle {{ width: 40px; height: 40px; border-radius: 12px; background: rgba(31,142,241,0.15); display: grid; place-items: center; font-weight: 700; color: var(--text); border: 1px solid var(--border); }}
    .links {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }}
    .metric-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    pre {{ background: var(--panel-2); padding: 8px; border-radius: 8px; border: 1px solid var(--border); overflow: auto; color: #a5d6ff; margin: 0; }}
    label {{ display: block; margin-top: 8px; font-size: 13px; color: var(--muted); }}
    input {{ width: 100%; padding: 8px; border-radius: 6px; border: 1px solid var(--border); background:#0f1928; color:#e6edf3; }}
    button {{ padding: 9px 12px; background: var(--accent); border: none; border-radius: 6px; color: white; cursor: pointer; }}
    button:hover {{ background: #1b7cd1; }}
    .hero {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }}
    .latest-block {{ display: grid; gap: 6px; }}
    .small {{ font-size: 12px; color: var(--muted); }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="logo">Oracle Lottery</div>
      <div class="sidebar-section">
        <div class="sidebar-title">Lotteries</div>
        <div id="lottery-list" class="lottery-list"></div>
      </div>
      <div class="sidebar-section">
        <div class="sidebar-title">Settings & Options</div>
        <button id="settings-link" class="sidebar-button">Open scheduler settings</button>
      </div>
    </aside>
    <main class="content">
      <section class="hero card">
        <div>
          <h1>Oracle Lottery Control Center</h1>
          <p class="muted" id="selected-description">Select a lottery to view latest official draws, jackpots, history readiness and prediction accuracy.</p>
        </div>
        <div class="chips">
          <span class="pill" id="history-pill">History: --</span>
          <span class="pill" id="jackpot-pill">Jackpot: --</span>
        </div>
      </section>

      <section class="grid">
        <div class="card">
          <div class="card-header">
            <h3>Latest draws & jackpots</h3>
            <span class="muted">Based on available history files</span>
          </div>
          <div id="news-grid" class="news-grid"></div>
        </div>
        <div class="card">
          <div class="card-header">
            <h3>Scheduler status</h3>
          </div>
          <pre id="status">Loading...</pre>
        </div>
      </section>

      <section class="grid">
        <div class="card">
          <div class="card-header">
            <div class="title-row">
              <div class="icon-circle" id="lottery-icon">🎯</div>
              <div>
                <h3 id="lottery-title">Select a lottery</h3>
                <div class="muted" id="lottery-country"></div>
              </div>
            </div>
            <button id="predict-btn">Request Prediction</button>
          </div>
          <p id="lottery-description" class="muted"></p>
          <div class="links">
            <a id="official-link" href="#" target="_blank" rel="noreferrer noopener">Official page</a>
            <a id="results-link" href="#" target="_blank" rel="noreferrer noopener">Results & history</a>
          </div>
        </div>
        <div class="card">
          <div class="card-header">
            <h3>Latest draw & prediction readiness</h3>
          </div>
          <div class="latest-block">
            <div class="muted small">Latest official draw</div>
            <div id="latest-draw">No draw data yet</div>
            <div class="muted small">Prediction statistics</div>
            <div id="prediction-stats" class="prediction-stats"></div>
          </div>
        </div>
      </section>

      <section class="grid">
        <div class="card">
          <div class="card-header">
            <h3>Notifications</h3>
          </div>
          <label for="email">Email</label>
          <input id="email" placeholder="user@example.com">
          <label for="phone">SMS</label>
          <input id="phone" placeholder="+1-555-123-4567">
          <button id="send-test">Send Test Alerts</button>
          <pre id="notifications">Waiting...</pre>
        </div>
        <div class="card" id="settings-card">
          <div class="card-header">
            <h3>Scheduler & Settings</h3>
          </div>
          <label for="interval">Default Interval (hours)</label>
          <input id="interval" type="number" min="1" max="720" value="12">
          <label><input id="enabled" type="checkbox" checked> Scheduler Enabled</label>
          <button id="save">Save</button>
          <div id="save-result" class="muted small"></div>
        </div>
      </section>

      <section class="grid">
        <div class="card">
          <div class="card-header">
            <h3>Metrics</h3>
          </div>
          <pre id="metrics">...</pre>
        </div>
      </section>
    </main>
  </div>
  <script>
    const state = {{
      lotteries: [],
      news: [],
      selectedId: null,
    }};

    async function fetchJSON(path) {{
      const res = await fetch(path).catch(() => null);
      if (!res) return null;
      return await res.json().catch(() => null);
    }}

    function renderSidebar() {{
      const list = document.getElementById('lottery-list');
      if (!list) return;
      list.innerHTML = '';
      state.lotteries.forEach((lotto) => {{
        const row = document.createElement('button');
        row.className = 'lotto-row' + (state.selectedId === lotto.id ? ' active' : '');
        row.innerHTML = `
          <span class="flag">${{lotto.flag_emoji || '🎯'}}</span>
          <div class="lotto-names">
            <strong>${{lotto.display_name}}</strong>
            <span class="lotto-country">${{lotto.country || 'N/A'}}</span>
          </div>
        `;
        row.onclick = () => selectLottery(lotto.id);
        list.appendChild(row);
      }});
    }}

    function renderNews() {{
      const grid = document.getElementById('news-grid');
      if (!grid) return;
      grid.innerHTML = '';
      if (!state.news.length) {{
        grid.innerHTML = '<div class="news-card muted">No draw news yet.</div>';
        return;
      }}
      state.news.forEach((item) => {{
        const card = document.createElement('div');
        card.className = 'news-card';
        const nums = (item.numbers || []).map((n) => `<span class="chip">${{n}}</span>`).join('');
        card.innerHTML = `
          <div class="muted small">${{item.date || 'Unknown date'}}</div>
          <strong>${{item.title}}</strong>
          <div class="chips">${{nums}}</div>
          <div class="chips">
            <span class="chip">${{item.has_jackpot ? 'Jackpot available' : 'No jackpot flag'}}</span>
            <span class="chip">${{item.history_available ? 'History ready' : 'History pending'}}</span>
          </div>
        `;
        grid.appendChild(card);
      }});
    }}

    function updateHeroPills(lotto) {{
      const historyPill = document.getElementById('history-pill');
      const jackpotPill = document.getElementById('jackpot-pill');
      if (historyPill) historyPill.textContent = `History: ${{lotto.history_available ? 'available' : 'missing'}}`;
      if (jackpotPill) jackpotPill.textContent = `Jackpot: ${{lotto.has_jackpot ? 'tracking' : 'n/a'}}`;
    }}

    function renderPrediction(lotto) {{
      const container = document.getElementById('prediction-stats');
      if (!container) return;
      const prediction = lotto.prediction || {{}};
      container.innerHTML = `
        <div class="metric-row">
          <div class="pill">Pending: ${{prediction.pending || 0}}</div>
          <div class="pill">Completed: ${{prediction.completed || 0}}</div>
        </div>
        <div class="muted small">Accuracy: ${{prediction.accuracy != null ? prediction.accuracy : 'Not yet recorded'}}</div>
      `;
    }}

    function renderLatestDraw(lotto) {{
      const el = document.getElementById('latest-draw');
      if (!el) return;
      const latest = lotto.latest_draw;
      if (!latest) {{
        el.textContent = 'No draw data yet';
        return;
      }}
      const chips = (latest.numbers || []).map((n) => `<span class="chip">${{n}}</span>`).join('');
      el.innerHTML = `
        <div class="muted small">Date: ${{latest.date || 'unknown'}}</div>
        <div class="chips">${{chips}}</div>
      `;
    }}

    function renderSelected(lotto) {{
      document.getElementById('lottery-title').textContent = lotto.display_name || 'Lottery';
      document.getElementById('lottery-country').textContent = `${{lotto.country || 'Unknown'}} • ${{lotto.flag_emoji || '🎯'}}`;
      document.getElementById('lottery-description').textContent = lotto.description || '';
      document.getElementById('official-link').href = lotto.official_url || '#';
      document.getElementById('results-link').href = lotto.results_url || '#';
      const icon = document.getElementById('lottery-icon');
      if (icon) {{
        icon.textContent = lotto.flag_emoji || '🎯';
      }}
      updateHeroPills(lotto);
      renderLatestDraw(lotto);
      renderPrediction(lotto);
      document.getElementById('selected-description').textContent = `Viewing ${{lotto.display_name}} — latest draws, jackpot flag and AI prediction readiness.`;
    }}

    function selectLottery(id) {{
      state.selectedId = id;
      const lotto = state.lotteries.find((l) => l.id === id);
      if (lotto) {{
        renderSelected(lotto);
        renderSidebar();
      }}
    }}

    async function loadLotteries() {{
      const payload = await fetchJSON('/api/lotteries');
      state.lotteries = payload?.lotteries || [];
      state.news = payload?.news || [];
      renderSidebar();
      renderNews();
      if (!state.selectedId && state.lotteries.length) {{
        selectLottery(state.lotteries[0].id);
      }} else if (state.selectedId) {{
        selectLottery(state.selectedId);
      }}
    }}

    async function refreshStatus() {{
      const status = await fetchJSON('/api/status');
      const pre = document.getElementById('status');
      if (pre) pre.textContent = JSON.stringify(status || {{}}, null, 2);
    }}

    async function refreshMetrics() {{
      const metrics = await fetchJSON('/api/metrics');
      const pre = document.getElementById('metrics');
      if (pre) pre.textContent = JSON.stringify(metrics || {{}}, null, 2);
    }}

    async function refreshNotifications() {{
      const notifications = await fetchJSON('/api/notifications');
      const pre = document.getElementById('notifications');
      if (pre) pre.textContent = JSON.stringify(notifications || {{}}, null, 2);
    }}

    async function refresh() {{
      await Promise.all([refreshStatus(), refreshMetrics(), refreshNotifications(), loadLotteries()]);
    }}

    document.getElementById('save').onclick = async () => {{
      const payload = {{
        default_interval_hours: Number(document.getElementById('interval').value),
        enabled: document.getElementById('enabled').checked
      }};
      const res = await fetch('/api/config', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      const body = await res.json().catch(() => ({{ status: 'error' }}));
      document.getElementById('save-result').textContent = JSON.stringify(body);
      refreshStatus();
    };

    document.getElementById('send-test').onclick = async () => {{
      const payload = {{
        email: document.getElementById('email').value,
        phone: document.getElementById('phone').value
      }};
      await fetch('/api/notify/test', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      refreshNotifications();
    }};

    document.getElementById('predict-btn').onclick = async () => {{
      if (!state.selectedId) return;
      await fetch(`/api/lotteries/${{state.selectedId}}/predict`, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ requested_at: Date.now() }})
      }}).catch(() => null);
      refreshMetrics();
    }};

    document.getElementById('settings-link').onclick = () => {{
      const panel = document.getElementById('settings-card');
      if (panel && panel.scrollIntoView) {{
        panel.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      }}
    }};

    refresh();
    setInterval(refresh, 7000);
  </script>
</body>
</html>"""
    return html.replace("{{", "{").replace("}}", "}").encode("utf-8")


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200):
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, payload: bytes, status: int = 200):
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _asset_response(handler: BaseHTTPRequestHandler, request_path: str):
    """Serve small static assets (icons/flags)."""
    rel = request_path[len("/assets/") :] if request_path.startswith("/assets/") else ""
    candidate = (ASSETS_ROOT / rel).resolve()
    if not str(candidate).startswith(str(ASSETS_ROOT.resolve())) or not candidate.exists():
        return _json_response(handler, {"error": "Asset not found"}, status=404)

    if candidate.suffix.lower() in {".png", ".bmp"}:
        mime = "image/png" if candidate.suffix.lower() == ".png" else "image/bmp"
    elif candidate.suffix.lower() in {".ico"}:
        mime = "image/x-icon"
    else:
        mime = "application/octet-stream"

    body = candidate.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", mime)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length > 1_000_000:  # 1MB guardrail
        handler.send_error(413, "Payload too large")
        return {}

    body = handler.rfile.read(length) if length else b"{}"
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return {}


def build_handler(state: DashboardState):
    """Factory that returns a request handler bound to the given state."""

    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # pragma: no cover - silence default logging
            LOGGER.info("%s - %s", self.address_string(), fmt % args)

        def do_GET(self):  # noqa: N802 - API requires this name
            if self.path == "/" or self.path.startswith("/index.html"):
                return _html_response(self, _index_html())
            if self.path.startswith("/assets/"):
                return _asset_response(self, self.path)
            if self.path.startswith("/api/status"):
                return _json_response(self, state.status())
            if self.path.startswith("/api/metrics"):
                return _json_response(self, state.metrics.summarize())
            if self.path.startswith("/api/notifications"):
                return _json_response(self, state.notifications.history())
            if self.path.startswith("/api/news"):
                overview = state.lotteries_overview()
                return _json_response(self, overview.get("news", []))
            if self.path.startswith("/api/lotteries"):
                overview = state.lotteries_overview()
                if self.path == "/api/lotteries":
                    return _json_response(self, overview)
                # Detail view: /api/lotteries/<id>
                parts = self.path.split("/")
                if len(parts) >= 4:
                    game = parts[3]
                    detail = next(
                        (entry for entry in overview["lotteries"] if entry["id"] == game),
                        {"error": "Not found"},
                    )
                    return _json_response(self, detail, status=200 if "error" not in detail else 404)

            return _json_response(self, {"error": "Not found"}, status=404)

        def do_POST(self):  # noqa: N802 - API requires this name
            if self.path.startswith("/api/config"):
                payload = _read_json(self)
                updated = state.update_config(payload)
                return _json_response(self, updated)

            if self.path.startswith("/api/notify/test"):
                payload = _read_json(self)
                email = payload.get("email", "")
                phone = payload.get("phone", "")
                result = {
                    "email": state.notifications.send_email(
                        email or "demo@example.com",
                        "Oracle Lottery test alert",
                        "Web dashboard test notification",
                    ),
                    "sms": state.notifications.send_sms(
                        phone or "+15550000000",
                        "Oracle Lottery dashboard test SMS",
                    ),
                }
                state.metrics.add_event("test_notification_sent")
                return _json_response(self, result)

            if self.path.startswith("/api/lotteries/") and self.path.endswith("/predict"):
                parts = self.path.split("/")
                if len(parts) >= 4:
                    game = parts[3]
                    state.metrics.add_event(f"prediction_request::{game}")
                    return _json_response(
                        self,
                        {"status": "queued", "game": game, "message": "Prediction request logged"},
                    )
                return _json_response(self, {"error": "Unknown lottery"}, status=400)

            return _json_response(self, {"error": "Not found"}, status=404)

    return DashboardHandler


def start_dashboard_server(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    scheduler: Optional["LotteryScheduler"] = None,
    data_root: Optional[Path] = None,
) -> ThreadingHTTPServer:
    """Start the dashboard server in a background thread."""
    state = DashboardState(scheduler=scheduler, data_root=data_root)
    server = ThreadingHTTPServer((host, port), build_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    LOGGER.info("Web dashboard running at http://%s:%s", host, port)
    return server


def main():
    parser = argparse.ArgumentParser(description="Start the Oracle Lottery web dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8080)")
    parser.add_argument("--data-root", default=None, help="Override data directory")
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Start dashboard without binding to LotteryScheduler (status will be limited)",
    )
    args = parser.parse_args()

    scheduler = None
    if not args.no_scheduler:
        try:
            from .lottery_scheduler import LotteryScheduler

            scheduler = LotteryScheduler(data_root=args.data_root)
        except Exception as exc:  # pragma: no cover - runtime convenience
            LOGGER.warning("Scheduler unavailable: %s", exc)

    data_root = Path(args.data_root) if args.data_root else None
    server = start_dashboard_server(
        host=args.host,
        port=args.port,
        scheduler=scheduler,
        data_root=data_root,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        LOGGER.info("Shutting down dashboard server")
        server.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
