"""
Unit tests for lottery scheduler module.
"""
import pytest
import os
import json
import sys
import tempfile
import shutil
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lottery_scheduler import LotteryScheduler


class TestLotteryScheduler:
    """Test lottery scheduler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary data directory
        self.test_data_root = tempfile.mkdtemp()
        self.scheduler = LotteryScheduler(data_root=self.test_data_root, use_apscheduler=False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.scheduler.running:
            self.scheduler.stop()
        if os.path.exists(self.test_data_root):
            shutil.rmtree(self.test_data_root)
    
    def test_scheduler_initialization(self):
        """Test that scheduler initializes correctly."""
        assert self.scheduler.fetcher is not None
        assert isinstance(self.scheduler.schedule_config, dict)
        assert self.scheduler.config_file.endswith('schedule_config.json')
    
    def test_default_config(self):
        """Test default configuration structure."""
        config = self.scheduler._get_default_config()
        
        assert "enabled" in config
        assert "default_interval_hours" in config
        assert "schedules" in config
        assert "global_settings" in config
        assert isinstance(config["schedules"], dict)
        assert isinstance(config["global_settings"], dict)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Modify config
        self.scheduler.schedule_config["enabled"] = True
        self.scheduler.schedule_config["schedules"]["TEST_LOTTERY"] = {
            "enabled": True,
            "interval_hours": 6
        }
        
        # Save
        self.scheduler._save_schedule_config()
        
        # Verify file exists
        assert os.path.exists(self.scheduler.config_file)
        
        # Load with new scheduler
        scheduler2 = LotteryScheduler(data_root=self.test_data_root, use_apscheduler=False)
        assert scheduler2.schedule_config["enabled"] is True
        assert "TEST_LOTTERY" in scheduler2.schedule_config["schedules"]
    
    def test_add_schedule(self):
        """Test adding a schedule for a lottery."""
        self.scheduler.add_schedule("TZOKER", interval_hours=8, enabled=True)
        
        assert "TZOKER" in self.scheduler.schedule_config["schedules"]
        schedule = self.scheduler.schedule_config["schedules"]["TZOKER"]
        assert schedule["enabled"] is True
        assert schedule["interval_hours"] == 8
    
    def test_add_schedule_with_custom_interval(self):
        """Test adding schedule with custom interval."""
        self.scheduler.add_schedule("EUROJACKPOT", interval_hours=24)
        
        schedule = self.scheduler.schedule_config["schedules"]["EUROJACKPOT"]
        assert schedule["interval_hours"] == 24
    
    def test_remove_schedule(self):
        """Test removing a schedule."""
        # Add then remove
        self.scheduler.add_schedule("TZOKER", interval_hours=12)
        assert "TZOKER" in self.scheduler.schedule_config["schedules"]
        
        self.scheduler.remove_schedule("TZOKER")
        assert "TZOKER" not in self.scheduler.schedule_config["schedules"]
    
    def test_enable_all_lotteries(self):
        """Test enabling all lotteries."""
        self.scheduler.enable_all_lotteries(interval_hours=6)
        
        assert self.scheduler.schedule_config["enabled"] is True
        # Should have schedules for all lotteries
        assert len(self.scheduler.schedule_config["schedules"]) > 0
        
        # Check one lottery has correct interval
        for game, config in self.scheduler.schedule_config["schedules"].items():
            assert config["interval_hours"] == 6
            assert config["enabled"] is True
            break  # Just check one
    
    def test_get_status(self):
        """Test getting scheduler status."""
        self.scheduler.enable_all_lotteries(interval_hours=12)
        status = self.scheduler.get_status()
        
        assert isinstance(status, dict)
        assert "enabled" in status
        assert "backend" in status
        assert "running" in status
        assert "scheduled_lotteries" in status
        assert isinstance(status["scheduled_lotteries"], list)
    
    def test_status_has_correct_backend(self):
        """Test that status reports correct backend."""
        status = self.scheduler.get_status()
        
        # Should be "Simple" since we disabled APScheduler
        assert status["backend"] == "Simple"
    
    def test_unknown_lottery_handling(self):
        """Test handling of unknown lottery in add_schedule."""
        # Should not crash, just log error
        self.scheduler.add_schedule("UNKNOWN_LOTTERY", interval_hours=12)
        
        # Should not be in schedules
        assert "UNKNOWN_LOTTERY" not in self.scheduler.schedule_config["schedules"]
    
    def test_config_persistence(self):
        """Test that configuration persists across instances."""
        # Set up schedule
        self.scheduler.add_schedule("TZOKER", interval_hours=8)
        self.scheduler.schedule_config["enabled"] = True
        self.scheduler._save_schedule_config()
        
        # Create new scheduler instance
        scheduler2 = LotteryScheduler(data_root=self.test_data_root)
        
        # Should load the same config
        assert scheduler2.schedule_config["enabled"] is True
        assert "TZOKER" in scheduler2.schedule_config["schedules"]
        assert scheduler2.schedule_config["schedules"]["TZOKER"]["interval_hours"] == 8
    
    def test_global_settings_default_values(self):
        """Test that global settings have expected default values."""
        config = self.scheduler._get_default_config()
        settings = config["global_settings"]
        
        assert settings["respect_cache"] is True
        assert settings["batch_delay_seconds"] == 2
        assert settings["max_retries"] == 3
        assert settings["retry_delay_minutes"] == 5
    
    def test_scheduler_not_running_initially(self):
        """Test that scheduler is not running on initialization."""
        assert self.scheduler.running is False
    
    def test_stop_when_not_running(self):
        """Test that stop() works even when not running."""
        # Should not crash
        self.scheduler.stop()
        assert self.scheduler.running is False


class TestSchedulerConfiguration:
    """Test configuration file handling."""
    
    def test_config_file_created_on_save(self):
        """Test that config file is created when saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = LotteryScheduler(data_root=tmpdir)
            scheduler._save_schedule_config()
            
            config_file = os.path.join(tmpdir, "schedule_config.json")
            assert os.path.exists(config_file)
    
    def test_config_file_is_valid_json(self):
        """Test that config file contains valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = LotteryScheduler(data_root=tmpdir)
            scheduler.add_schedule("TZOKER", interval_hours=12)
            scheduler._save_schedule_config()
            
            config_file = os.path.join(tmpdir, "schedule_config.json")
            with open(config_file, 'r') as f:
                data = json.load(f)  # Should not raise
            
            assert isinstance(data, dict)
            assert "schedules" in data
    
    def test_corrupted_config_falls_back_to_default(self):
        """Test that corrupted config file falls back to defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "schedule_config.json")
            
            # Create corrupted JSON file
            os.makedirs(tmpdir, exist_ok=True)
            with open(config_file, 'w') as f:
                f.write("{ invalid json }")
            
            # Should fall back to defaults without crashing
            scheduler = LotteryScheduler(data_root=tmpdir)
            assert isinstance(scheduler.schedule_config, dict)
            assert "schedules" in scheduler.schedule_config


class TestSchedulerIntegration:
    """Integration tests for scheduler."""
    
    def test_scheduler_with_fetcher_integration(self):
        """Test that scheduler properly integrates with fetcher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = LotteryScheduler(data_root=tmpdir)
            
            # Fetcher should be initialized
            assert scheduler.fetcher is not None
            assert scheduler.fetcher.data_root == tmpdir
    
    def test_enable_all_creates_all_schedules(self):
        """Test that enable_all creates schedules for all known lotteries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = LotteryScheduler(data_root=tmpdir)
            scheduler.enable_all_lotteries(interval_hours=12)
            
            # Should have schedules for all games
            from ultra_lottery_helper import GAMES
            for game in GAMES.keys():
                assert game in scheduler.schedule_config["schedules"]
