"""
Integration tests for data loading functionality.
Tests CSV/Excel loading, validation, and history merging.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ultra_lottery_helper import (
    _load_all_history,
    GAMES,
    _game_path,
)


class TestDataLoading:
    """Test data loading from various sources."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_load_empty_directory(self, temp_data_dir, monkeypatch):
        """Test loading from empty directory."""
        # Mock the game path to use temp directory
        monkeypatch.setattr(
            "ultra_lottery_helper._game_path",
            lambda game: temp_data_dir
        )
        df, msg = _load_all_history("LOTTO", use_online=False)
        assert df.empty
        assert "No valid data loaded" in msg
    
    def test_load_valid_csv(self, temp_data_dir, monkeypatch):
        """Test loading valid CSV file."""
        # Create a valid CSV file
        csv_path = os.path.join(temp_data_dir, "test.csv")
        test_data = pd.DataFrame({
            "n1": [1, 5, 10],
            "n2": [2, 6, 11],
            "n3": [3, 7, 12],
            "n4": [4, 8, 13],
            "n5": [5, 9, 14],
            "n6": [6, 10, 15],
            "date": pd.date_range("2020-01-01", periods=3),
        })
        test_data.to_csv(csv_path, index=False)
        
        # Mock the game path
        monkeypatch.setattr(
            "ultra_lottery_helper._game_path",
            lambda game: temp_data_dir
        )
        
        df, msg = _load_all_history("LOTTO", use_online=False)
        assert not df.empty
        assert len(df) == 3
        assert "3 valid draws" in msg
    
    def test_load_multiple_files(self, temp_data_dir, monkeypatch):
        """Test loading and merging multiple files."""
        # Create multiple CSV files
        for i in range(3):
            csv_path = os.path.join(temp_data_dir, f"test_{i}.csv")
            test_data = pd.DataFrame({
                "n1": [1 + i, 5 + i],
                "n2": [2 + i, 6 + i],
                "n3": [3 + i, 7 + i],
                "n4": [4 + i, 8 + i],
                "n5": [5 + i, 9 + i],
                "n6": [6 + i, 10 + i],
                "date": pd.date_range(f"2020-0{i+1}-01", periods=2),
            })
            test_data.to_csv(csv_path, index=False)
        
        monkeypatch.setattr(
            "ultra_lottery_helper._game_path",
            lambda game: temp_data_dir
        )
        
        df, msg = _load_all_history("LOTTO", use_online=False)
        assert not df.empty
        assert len(df) == 6  # 2 rows per file * 3 files
    
    def test_load_invalid_data(self, temp_data_dir, monkeypatch):
        """Test handling of invalid data."""
        # Create CSV with invalid number ranges
        csv_path = os.path.join(temp_data_dir, "invalid.csv")
        test_data = pd.DataFrame({
            "n1": [100, 200],  # Invalid: > max
            "n2": [2, 6],
            "n3": [3, 7],
            "n4": [4, 8],
            "n5": [5, 9],
            "n6": [6, 10],
            "date": pd.date_range("2020-01-01", periods=2),
        })
        test_data.to_csv(csv_path, index=False)
        
        monkeypatch.setattr(
            "ultra_lottery_helper._game_path",
            lambda game: temp_data_dir
        )
        
        df, msg = _load_all_history("LOTTO", use_online=False)
        # Invalid rows should be filtered out
        assert len(df) == 0
    
    def test_load_missing_columns(self, temp_data_dir, monkeypatch):
        """Test handling of missing required columns."""
        csv_path = os.path.join(temp_data_dir, "missing_cols.csv")
        test_data = pd.DataFrame({
            "n1": [1, 5],
            "n2": [2, 6],
            # Missing n3, n4, n5, n6
            "date": pd.date_range("2020-01-01", periods=2),
        })
        test_data.to_csv(csv_path, index=False)
        
        monkeypatch.setattr(
            "ultra_lottery_helper._game_path",
            lambda game: temp_data_dir
        )
        
        df, msg = _load_all_history("LOTTO", use_online=False)
        assert df.empty
        assert "Missing columns" in msg or "No valid data" in msg
    
    def test_load_sorts_numbers(self, temp_data_dir, monkeypatch):
        """Test that numbers are sorted within each draw."""
        csv_path = os.path.join(temp_data_dir, "unsorted.csv")
        # Create data with unsorted numbers
        test_data = pd.DataFrame({
            "n1": [10, 20],
            "n2": [5, 15],
            "n3": [15, 10],
            "n4": [1, 5],
            "n5": [20, 25],
            "n6": [25, 30],
            "date": pd.date_range("2020-01-01", periods=2),
        })
        test_data.to_csv(csv_path, index=False)
        
        monkeypatch.setattr(
            "ultra_lottery_helper._game_path",
            lambda game: temp_data_dir
        )
        
        df, msg = _load_all_history("LOTTO", use_online=False)
        assert not df.empty
        # Check that numbers in each row are sorted
        for idx in range(len(df)):
            row_nums = df.iloc[idx][["n1", "n2", "n3", "n4", "n5", "n6"]].values
            assert np.array_equal(row_nums, np.sort(row_nums))


class TestDataValidation:
    """Test data validation logic."""
    
    def test_number_range_validation(self):
        """Test that numbers are validated against game specs."""
        spec = GAMES["LOTTO"]
        assert spec.main_max == 49
        # Numbers should be between 1 and 49 for LOTTO
    
    def test_tzoker_validation(self):
        """Test TZOKER-specific validation."""
        spec = GAMES["TZOKER"]
        assert spec.main_max == 45
        assert spec.sec_max == 20
        # Main numbers: 1-45, Joker: 1-20
    
    def test_eurojackpot_validation(self):
        """Test EUROJACKPOT-specific validation."""
        spec = GAMES["EUROJACKPOT"]
        assert spec.main_max == 50
        assert spec.sec_max == 12
        # Main numbers: 1-50, Euro numbers: 1-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
