"""
Tests for the AI/IA learning system (ulh_learning.py)
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


class TestLearningSystemImports:
    """Test that the learning system modules can be imported correctly."""
    
    def test_learning_module_import(self):
        """Test that ulh_learning module can be imported."""
        from ulh_learning import (
            record_portfolio, 
            record_outcome, 
            learn_after_draw,
            get_status_summary,
            apply_state_to_config
        )
        assert callable(record_portfolio)
        assert callable(record_outcome)
        assert callable(learn_after_draw)
        assert callable(get_status_summary)
        assert callable(apply_state_to_config)
    
    def test_learning_cli_import(self):
        """Test that ulh_learn_cli module can be imported."""
        from ulh_learn_cli import main, parse_combo
        assert callable(main)
        assert callable(parse_combo)


class TestLearningSystemFunctionality:
    """Test basic learning system functionality."""
    
    def test_get_status_summary(self):
        """Test getting learning system status."""
        from ulh_learning import get_status_summary
        
        status = get_status_summary()
        assert isinstance(status, dict)
        assert "state" in status
        assert "report" in status
        
        # Check state structure
        state = status["state"]
        assert "luck_beta" in state
        assert "unluck_gamma" in state
        assert "half_life" in state
        assert "ensemble" in state
        
        # Check ensemble weights
        ensemble = state["ensemble"]
        assert "ewma" in ensemble
        assert "recency" in ensemble
        assert "ml" in ensemble
    
    def test_apply_state_to_config(self):
        """Test applying learning state to config."""
        from ulh_learning import apply_state_to_config, _load_state
        from ultra_lottery_helper import Config
        
        cfg = Config()
        original_luck_beta = cfg.luck_beta
        
        # Apply state
        cfg = apply_state_to_config(cfg)
        
        # Config should have state values applied
        state = _load_state()
        assert cfg.luck_beta == state["luck_beta"]
        assert cfg.unluck_gamma == state["unluck_gamma"]
        assert cfg.half_life == state["half_life"]
    
    def test_record_portfolio(self):
        """Test recording a prediction portfolio."""
        from ulh_learning import record_portfolio
        
        # Use a temporary database for testing
        game = "TZOKER"
        portfolio = [[1, 5, 12, 27, 38], [3, 14, 22, 33, 41]]
        tag = "test_portfolio"
        
        # Record portfolio
        n = record_portfolio(game, portfolio, tag=tag)
        assert n == 2
    
    def test_record_outcome(self):
        """Test recording a draw outcome."""
        from ulh_learning import record_outcome
        
        game = "TZOKER"
        main = [3, 14, 22, 33, 41]
        sec = [5]
        
        # Record outcome
        n = record_outcome(game, main, sec)
        assert n == 1
    
    def test_parse_combo(self):
        """Test combo parsing from CLI."""
        from ulh_learn_cli import parse_combo
        
        # Test space-separated
        combo1 = parse_combo("1 5 12 27 38")
        assert combo1 == [1, 5, 12, 27, 38]
        
        # Test comma-separated
        combo2 = parse_combo("1,5,12,27,38")
        assert combo2 == [1, 5, 12, 27, 38]
        
        # Test mixed
        combo3 = parse_combo("1, 5, 12 27,38")
        assert combo3 == [1, 5, 12, 27, 38]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
