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
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

try:  # Support both package and script execution
    from .utils import get_logger, load_json, save_json
except ImportError:  # pragma: no cover - fallback for direct execution
    from utils import get_logger, load_json, save_json  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from .lottery_scheduler import LotteryScheduler

LOGGER = get_logger("web_dashboard")
DEFAULT_PORT = 8080


def _default_data_root() -> Path:
    """Return the default data root path."""
    return Path(__file__).resolve().parent / "data"


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
    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Oracle Lottery Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #0b1521; color: #e6edf3; }}
    h1 {{ margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ background: #132235; border: 1px solid #1f3b57; border-radius: 8px; padding: 12px; }}
    pre {{ background: #0f1928; padding: 8px; overflow: auto; color: #a5d6ff; }}
    label {{ display: block; margin-top: 8px; }}
    input {{ width: 100%; padding: 6px; border-radius: 4px; border: 1px solid #1f3b57; background:#0f1928; color:#e6edf3; }}
    button {{ margin-top: 8px; padding: 8px 12px; background: #1f8ef1; border: none; border-radius: 4px; color: white; cursor: pointer; }}
    button:hover {{ background: #1b7cd1; }}
  </style>
</head>
<body>
  <h1>Oracle Lottery Web Dashboard</h1>
  <div class="grid">
    <div class="card">
      <h3>Real-time Status</h3>
      <pre id="status">Loading...</pre>
    </div>
    <div class="card">
      <h3>Update Scheduler</h3>
      <label for="interval">Default Interval (hours)</label>
      <input id="interval" type="number" min="1" max="720" value="12">
      <label><input id="enabled" type="checkbox" checked> Scheduler Enabled</label>
      <button id="save">Save</button>
      <div id="save-result"></div>
    </div>
    <div class="card">
      <h3>Notifications</h3>
      <label for="email">Email</label>
      <input id="email" placeholder="user@example.com">
      <label for="phone">SMS</label>
      <input id="phone" placeholder="+1-555-123-4567">
      <button id="send-test">Send Test Alerts</button>
      <pre id="notifications">Waiting...</pre>
    </div>
    <div class="card">
      <h3>Metrics</h3>
      <pre id="metrics">...</pre>
    </div>
  </div>
  <script>
    async function fetchJSON(path) {{
      const res = await fetch(path).catch(() => null);
      if (!res) return null;
      return await res.json();
    }}

    async function refresh() {{
      const status = await fetchJSON('/api/status');
      document.getElementById('status').textContent = JSON.stringify(status || {{}}, null, 2);
      const metrics = await fetchJSON('/api/metrics');
      document.getElementById('metrics').textContent = JSON.stringify(metrics || {{}}, null, 2);
      const notifications = await fetchJSON('/api/notifications');
      document.getElementById('notifications').textContent = JSON.stringify(notifications || {{}}, null, 2);
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
      refresh();
    }};

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
      refresh();
    }};

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>""".encode("utf-8")


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
            if self.path.startswith("/api/status"):
                return _json_response(self, state.status())
            if self.path.startswith("/api/metrics"):
                return _json_response(self, state.metrics.summarize())
            if self.path.startswith("/api/notifications"):
                return _json_response(self, state.notifications.history())

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
