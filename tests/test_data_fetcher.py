"""
Unit tests for lottery data fetcher module.
"""
import pytest
import os
import json
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lottery_data_fetcher import LotteryDataFetcher, fetch_latest_draws
from ultra_lottery_helper import GAMES, fetch_online_history, LOTTERY_METADATA


class TestLotteryDataFetcher:
    """Test lottery data fetcher functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary data directory
        self.test_data_root = tempfile.mkdtemp()
        self.fetcher = LotteryDataFetcher(data_root=self.test_data_root)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_data_root):
            shutil.rmtree(self.test_data_root)
    
    def test_fetch_log_initialization(self):
        """Test that fetch log is initialized correctly."""
        assert isinstance(self.fetcher.fetch_log, dict)
        assert self.fetcher.fetch_log_file.endswith('fetch_log.json')
    
    def test_save_and_load_fetch_log(self):
        """Test saving and loading fetch log."""
        # Add test data
        self.fetcher.fetch_log['TEST_LOTTERY'] = {
            'last_fetch': datetime.now().isoformat(),
            'rows_fetched': 100,
            'status': 'success'
        }
        
        # Save
        self.fetcher._save_fetch_log()
        
        # Verify file exists
        assert os.path.exists(self.fetcher.fetch_log_file)
        
        # Load with new fetcher instance
        fetcher2 = LotteryDataFetcher(data_root=self.test_data_root)
        assert 'TEST_LOTTERY' in fetcher2.fetch_log
        assert fetcher2.fetch_log['TEST_LOTTERY']['rows_fetched'] == 100
    
    def test_get_fetch_status(self):
        """Test getting fetch status as DataFrame."""
        status_df = self.fetcher.get_fetch_status()
        
        assert isinstance(status_df, pd.DataFrame)
        assert len(status_df) > 0  # Should have at least some lotteries
        assert 'Lottery' in status_df.columns
        assert 'Country' in status_df.columns
        assert 'Last Fetch' in status_df.columns
        assert 'Has Jackpot' in status_df.columns
    
    def test_fetch_status_columns(self):
        """Test that fetch status has expected columns."""
        status_df = self.fetcher.get_fetch_status()
        
        expected_cols = ['Lottery', 'Country', 'Last Fetch', 'Rows', 'Status', 'Has Jackpot']
        for col in expected_cols:
            assert col in status_df.columns, f"Missing column: {col}"
    
    def test_fetch_status_never_fetched(self):
        """Test that unfetched lotteries show 'Never' status."""
        status_df = self.fetcher.get_fetch_status()
        
        # All should show "Never" initially
        assert all(status_df['Last Fetch'] == 'Never')
        assert all(status_df['Status'] == 'Not fetched')
    
    def test_unknown_lottery_handling(self):
        """Test handling of unknown lottery name."""
        success, msg = self.fetcher.fetch_lottery_data('UNKNOWN_LOTTERY')
        
        assert success is False
        assert 'Unknown lottery' in msg

    def test_unknown_lottery_raises_when_requested(self):
        """Ensure raise_on_unknown surfaces ValueError for invalid lotteries."""
        with pytest.raises(ValueError):
            self.fetcher.fetch_lottery_data('UNKNOWN_LOTTERY', raise_on_unknown=True)
    
    def test_fetch_log_structure(self):
        """Test that fetch log has correct structure after manual update."""
        # Manually add a fetch log entry
        test_entry = {
            'last_fetch': '2026-01-19T12:00:00',
            'rows_fetched': 150,
            'file': 'test.csv',
            'status': 'success'
        }
        
        self.fetcher.fetch_log['TZOKER'] = test_entry
        self.fetcher._save_fetch_log()
        
        # Reload
        fetcher2 = LotteryDataFetcher(data_root=self.test_data_root)
        
        assert 'TZOKER' in fetcher2.fetch_log
        assert fetcher2.fetch_log['TZOKER']['rows_fetched'] == 150
        assert fetcher2.fetch_log['TZOKER']['status'] == 'success'

    def test_api_integration_uses_json_payload(self, monkeypatch):
        """Ensure API JSON is preferred when available."""
        payload = {
            "draws": [
                {"date": "2020-01-01", "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "joker": 6}
            ]
        }

        class DummyResponse:
            def __init__(self, data):
                self._data = data

            def raise_for_status(self):
                return None

            def json(self):
                return self._data

        monkeypatch.setattr("ultra_lottery_helper.requests.get", lambda *a, **k: DummyResponse(payload))
        tzoker_meta = LOTTERY_METADATA.setdefault("TZOKER", {})
        monkeypatch.setitem(tzoker_meta, "api_endpoint", "https://api.example.com/tzoker")
        df, msg = fetch_online_history("TZOKER")
        assert not df.empty
        assert "API" in msg

    def test_email_notification_on_failure(self, monkeypatch):
        """Send email notification when fetch fails."""
        monkeypatch.setattr("lottery_data_fetcher.fetch_online_history", lambda game: (pd.DataFrame(), "network down"))
        fetcher = LotteryDataFetcher(data_root=self.test_data_root, alert_email="alerts@example.com")
        success, msg = fetcher.fetch_lottery_data("TZOKER", force=True)
        assert success is False
        log_path = Path(self.test_data_root) / "notifications.json"
        assert log_path.exists()

    def test_webhook_triggered_on_success(self, monkeypatch):
        """Post webhook after successful fetch."""
        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append({"url": url, "json": json})

            class Resp:
                status_code = 200

            return Resp()

        fixed_date = datetime(2020, 1, 1)
        df = pd.DataFrame(
            [{"date": fixed_date, "n1": 1, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "joker": 6}]
        )
        monkeypatch.setattr("lottery_data_fetcher.fetch_online_history", lambda game: (df, "ok"))
        fetcher = LotteryDataFetcher(data_root=self.test_data_root, webhook_urls="https://webhook.example.com")
        fetcher._http_post = fake_post
        success, _ = fetcher.fetch_lottery_data("TZOKER", force=True)
        assert success is True
        assert calls and calls[0]["url"] == "https://webhook.example.com"

    def test_anomaly_detection_blocks_bad_data(self, monkeypatch):
        """Detect out-of-range numbers and block save."""
        fixed_date = datetime(2020, 1, 1)
        bad_df = pd.DataFrame(
            [{"date": fixed_date, "n1": 99, "n2": 2, "n3": 3, "n4": 4, "n5": 5, "joker": 6}]
        )
        monkeypatch.setattr("lottery_data_fetcher.fetch_online_history", lambda game: (bad_df, "ok"))
        fetcher = LotteryDataFetcher(data_root=self.test_data_root, alert_email="alerts@example.com")
        success, msg = fetcher.fetch_lottery_data("TZOKER", force=True)
        assert success is False
        assert "Anomaly" in msg


class TestFetchLatestDraws:
    """Test the convenience function."""
    
    def test_fetch_latest_draws_returns_dict(self):
        """Test that fetch_latest_draws returns a dictionary."""
        # This will fail to fetch due to network, but should return dict structure
        result = fetch_latest_draws(game='TZOKER')
        
        assert isinstance(result, dict)
        assert 'TZOKER' in result
        assert 'success' in result['TZOKER']
        assert 'message' in result['TZOKER']


class TestIntegration:
    """Integration tests for data fetcher."""
    
    def test_fetcher_creates_data_directory(self):
        """Test that fetcher creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = LotteryDataFetcher(data_root=tmpdir)
            
            # Save fetch log should create directory
            fetcher._save_fetch_log()
            
            assert os.path.exists(tmpdir)
    
    def test_fetch_log_json_format(self):
        """Test that fetch log is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = LotteryDataFetcher(data_root=tmpdir)
            
            # Add test entry
            fetcher.fetch_log['TEST'] = {
                'last_fetch': datetime.now().isoformat(),
                'rows_fetched': 50,
                'status': 'success'
            }
            fetcher._save_fetch_log()
            
            # Read and parse JSON
            with open(fetcher.fetch_log_file, 'r') as f:
                data = json.load(f)
            
            assert 'TEST' in data
            assert data['TEST']['rows_fetched'] == 50

    def test_fetch_all_uses_stubbed_fetcher(self, monkeypatch):
        """Ensure fetch_all iterates through games concurrently-safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = LotteryDataFetcher(data_root=tmpdir, max_workers=2)
            calls = []

            def fake_fetch(game, force=False):
                calls.append(game)
                return True, f"{game} ok"

            monkeypatch.setattr(fetcher, "fetch_lottery_data", fake_fetch)
            results = fetcher.fetch_all_lotteries(force=True)
            assert set(calls) == set(GAMES.keys())
            assert set(results.keys()) == set(GAMES.keys())
