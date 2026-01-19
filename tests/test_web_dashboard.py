import tempfile
from pathlib import Path
import importlib.util
import sys

MODULE_PATH = Path(__file__).resolve().parent.parent / "src" / "web_dashboard.py"
sys.path.insert(0, str(MODULE_PATH.parent))
spec = importlib.util.spec_from_file_location("web_dashboard", MODULE_PATH)
web_dashboard = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(web_dashboard)  # type: ignore

DashboardState = web_dashboard.DashboardState
MetricsCollector = web_dashboard.MetricsCollector
NotificationManager = web_dashboard.NotificationManager


class FakeScheduler:
    def __init__(self):
        self.schedule_config = {
            "enabled": False,
            "default_interval_hours": 12,
            "global_settings": {},
        }
        self.saved = False

    def get_status(self):
        return {
            "enabled": self.schedule_config["enabled"],
            "backend": "Fake",
            "running": False,
            "scheduled_lotteries": [],
        }

    def _save_schedule_config(self):
        self.saved = True


def test_metrics_persist_to_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = MetricsCollector(data_root=Path(tmpdir))
        collector.record("fetch_success", 2)
        collector.add_event("ok")

        # Reload to ensure persistence
        collector2 = MetricsCollector(data_root=Path(tmpdir))
        summary = collector2.summarize()
        assert summary["counters"]["fetch_success"] == 2
        assert summary["events"]


def test_notification_history_and_logging():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = NotificationManager(data_root=Path(tmpdir))
        manager.send_email("user@example.com", "Hello", "Body")
        manager.send_sms("+1-555-123-4567", "Ping")

        history = manager.history()
        assert history["count"] == 2
        channels = {entry["channel"] for entry in history["recent"]}
        assert channels == {"email", "sms"}


def test_dashboard_state_updates_config_when_scheduler_present():
    fake = FakeScheduler()
    state = DashboardState(scheduler=fake, data_root=Path(tempfile.mkdtemp()))

    updated = state.update_config({"default_interval_hours": 6, "enabled": True})

    assert updated["default_interval_hours"] == 6
    assert updated["enabled"] is True
    assert fake.saved is True


def test_dashboard_state_status_includes_metrics_and_notifications():
    with tempfile.TemporaryDirectory() as tmpdir:
        state = DashboardState(
            scheduler=None,
            data_root=Path(tmpdir),
        )
        state.metrics.record("heartbeat")
        state.notifications.send_email("demo@example.com", "Hi", "Test")

        status = state.status()
        assert "metrics" in status
        assert "notifications" in status
        assert status["metrics"]["counters"]["heartbeat"] == 1
        assert status["notifications"]["recent"]
