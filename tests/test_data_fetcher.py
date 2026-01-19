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
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lottery_data_fetcher import LotteryDataFetcher, fetch_latest_draws


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
