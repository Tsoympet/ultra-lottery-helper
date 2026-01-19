"""
Unit tests for core lottery prediction logic.
Tests EWMA calculations, BMA, constraints, and statistical functions.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ultra_lottery_helper import (
    _ewma_weights,
    Config,
    GameSpec,
    GAMES,
    _luck_vectors,
    _rng,
)


class TestEWMACalculations:
    """Test EWMA weight calculations."""
    
    def test_ewma_weights_basic(self):
        """Test basic EWMA weight calculation."""
        weights = _ewma_weights(10, half_life=5)
        assert len(weights) == 10
        assert np.isclose(weights.sum(), 1.0)
        # Weights should be monotonically increasing (recent draws have more weight)
        assert all(weights[i] <= weights[i+1] for i in range(len(weights)-1))
    
    def test_ewma_weights_empty(self):
        """Test EWMA with empty input."""
        weights = _ewma_weights(0, half_life=5)
        assert len(weights) == 0
    
    def test_ewma_weights_single(self):
        """Test EWMA with single element."""
        weights = _ewma_weights(1, half_life=5)
        assert len(weights) == 1
        assert np.isclose(weights[0], 1.0)
    
    def test_ewma_weights_different_halflife(self):
        """Test EWMA with different half-life values."""
        w1 = _ewma_weights(10, half_life=5)
        w2 = _ewma_weights(10, half_life=10)
        # Longer half-life should give more uniform weights
        assert w1[-1] > w2[-1]  # Most recent weight is higher with shorter half-life


class TestConfig:
    """Test configuration and constraint settings."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        cfg = Config()
        assert cfg.iterations == 50000
        assert cfg.topk == 200
        assert cfg.seed == 42
        assert cfg.use_bma is True
        assert cfg.use_ewma is True
    
    def test_config_custom_values(self):
        """Test custom configuration values."""
        cfg = Config(
            iterations=10000,
            topk=100,
            seed=123,
            use_ml=True
        )
        assert cfg.iterations == 10000
        assert cfg.topk == 100
        assert cfg.seed == 123
        assert cfg.use_ml is True
    
    def test_adaptive_constraints_with_minimal_data(self):
        """Test adaptive constraints with minimal data (should skip)."""
        cfg = Config()
        spec = GAMES["LOTTO"]
        df = pd.DataFrame({
            "n1": [1, 2, 3],
            "n2": [4, 5, 6],
            "n3": [7, 8, 9],
            "n4": [10, 11, 12],
            "n5": [13, 14, 15],
            "n6": [16, 17, 18],
        })
        # Should not modify constraints with < 30 rows
        original_min_even = cfg.min_even
        cfg.set_adaptive_constraints(df, spec)
        assert cfg.min_even == original_min_even
    
    def test_adaptive_constraints_with_sufficient_data(self):
        """Test adaptive constraints with sufficient historical data."""
        cfg = Config()
        spec = GAMES["LOTTO"]
        # Create 50 sample draws
        np.random.seed(42)
        data = {
            "n1": np.random.randint(1, 10, 50),
            "n2": np.random.randint(11, 20, 50),
            "n3": np.random.randint(21, 30, 50),
            "n4": np.random.randint(31, 40, 50),
            "n5": np.random.randint(41, 48, 50),
            "n6": np.random.randint(1, 49, 50),
        }
        df = pd.DataFrame(data)
        cfg.set_adaptive_constraints(df, spec)
        # Constraints should be updated
        assert cfg.min_even >= 0
        assert cfg.max_even <= 6
        assert cfg.sum_min > 0
        assert cfg.sum_max > cfg.sum_min


class TestGameSpecs:
    """Test game specifications."""
    
    def test_tzoker_spec(self):
        """Test TZOKER game specification."""
        spec = GAMES["TZOKER"]
        assert spec.name == "TZOKER"
        assert spec.main_pick == 5
        assert spec.main_max == 45
        assert spec.sec_pick == 1
        assert spec.sec_max == 20
        assert "joker" in spec.cols
    
    def test_lotto_spec(self):
        """Test LOTTO game specification."""
        spec = GAMES["LOTTO"]
        assert spec.name == "LOTTO"
        assert spec.main_pick == 6
        assert spec.main_max == 49
        assert spec.sec_pick == 0
        assert len(spec.cols) == 6
    
    def test_eurojackpot_spec(self):
        """Test EUROJACKPOT game specification."""
        spec = GAMES["EUROJACKPOT"]
        assert spec.name == "EUROJACKPOT"
        assert spec.main_pick == 5
        assert spec.main_max == 50
        assert spec.sec_pick == 2
        assert spec.sec_max == 12
        assert "e1" in spec.cols and "e2" in spec.cols


class TestLuckVectors:
    """Test luck/unluck vector calculations."""
    
    def test_luck_vectors_basic(self):
        """Test basic luck vector calculation."""
        spec = GAMES["LOTTO"]
        df = pd.DataFrame({
            "n1": [1, 5, 10],
            "n2": [2, 6, 11],
            "n3": [3, 7, 12],
            "n4": [4, 8, 13],
            "n5": [5, 9, 14],
            "n6": [6, 10, 15],
        })
        drought, recent = _luck_vectors(df, "LOTTO", spec)
        assert len(drought) == spec.main_max
        assert len(recent) == spec.main_max
        # Values should be normalized between 0 and 1
        assert drought.min() >= 0 and drought.max() <= 1
        assert recent.min() >= 0 and recent.max() <= 1
    
    def test_luck_vectors_empty(self):
        """Test luck vectors with empty dataframe."""
        spec = GAMES["LOTTO"]
        df = pd.DataFrame({
            "n1": [], "n2": [], "n3": [], "n4": [], "n5": [], "n6": []
        })
        drought, recent = _luck_vectors(df, "LOTTO", spec)
        assert len(drought) == spec.main_max
        assert len(recent) == spec.main_max
        # Should be all zeros for empty data
        assert np.allclose(drought, 0.0)
        assert np.allclose(recent, 0.0)


class TestRNG:
    """Test random number generator."""
    
    def test_rng_with_seed(self):
        """Test RNG with fixed seed produces consistent results."""
        rng1 = _rng(42)
        rng2 = _rng(42)
        vals1 = rng1.random(10)
        vals2 = rng2.random(10)
        assert np.allclose(vals1, vals2)
    
    def test_rng_without_seed(self):
        """Test RNG without seed uses default."""
        rng1 = _rng(None)
        rng2 = _rng(None)
        # Both should use seed 42 by default
        vals1 = rng1.random(10)
        vals2 = rng2.random(10)
        assert np.allclose(vals1, vals2)
    
    def test_rng_different_seeds(self):
        """Test RNG with different seeds produces different results."""
        rng1 = _rng(42)
        rng2 = _rng(123)
        vals1 = rng1.random(10)
        vals2 = rng2.random(10)
        assert not np.allclose(vals1, vals2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
