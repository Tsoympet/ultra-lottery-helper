"""
Unit tests for European lottery specifications.
Tests that all new European lotteries are properly configured with correct game rules.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ultra_lottery_helper import (
    GAMES,
    OPAP_TICKET_PRICE_DEFAULTS,
    LOTTERY_METADATA,
    GameSpec,
)


class TestEuropeanLotterySpecs:
    """Test specifications for all European lotteries."""
    
    def test_all_lotteries_have_specs(self):
        """Test that all lotteries have game specifications."""
        expected_lotteries = [
            "TZOKER", "LOTTO", "EUROJACKPOT",
            "UK_NATIONAL_LOTTERY", "LA_PRIMITIVA", "SUPERENALOTTO",
            "LOTO_FRANCE", "LOTTO_6AUS49", "AUSTRIAN_LOTTO", "SWISS_LOTTO"
        ]
        for lottery in expected_lotteries:
            assert lottery in GAMES, f"{lottery} missing from GAMES"
            assert isinstance(GAMES[lottery], GameSpec)
    
    def test_uk_national_lottery_spec(self):
        """Test UK National Lottery specification."""
        spec = GAMES["UK_NATIONAL_LOTTERY"]
        assert spec.name == "UK_NATIONAL_LOTTERY"
        assert spec.main_pick == 6
        assert spec.main_max == 59
        assert spec.sec_pick == 0
        assert spec.sec_max == 0
        assert spec.cols == ["n1","n2","n3","n4","n5","n6"]
    
    def test_la_primitiva_spec(self):
        """Test La Primitiva specification."""
        spec = GAMES["LA_PRIMITIVA"]
        assert spec.name == "LA_PRIMITIVA"
        assert spec.main_pick == 6
        assert spec.main_max == 49
        assert spec.sec_pick == 1
        assert spec.sec_max == 9
        assert spec.cols == ["n1","n2","n3","n4","n5","n6","bonus"]
    
    def test_superenalotto_spec(self):
        """Test SuperEnalotto specification."""
        spec = GAMES["SUPERENALOTTO"]
        assert spec.name == "SUPERENALOTTO"
        assert spec.main_pick == 6
        assert spec.main_max == 90  # Italian lottery has 90 numbers
        assert spec.sec_pick == 0
        assert spec.sec_max == 0
        assert spec.cols == ["n1","n2","n3","n4","n5","n6"]
    
    def test_loto_france_spec(self):
        """Test Loto France specification."""
        spec = GAMES["LOTO_FRANCE"]
        assert spec.name == "LOTO_FRANCE"
        assert spec.main_pick == 5
        assert spec.main_max == 49
        assert spec.sec_pick == 1
        assert spec.sec_max == 10
        assert spec.cols == ["n1","n2","n3","n4","n5","chance"]
    
    def test_lotto_6aus49_spec(self):
        """Test German Lotto 6aus49 specification."""
        spec = GAMES["LOTTO_6AUS49"]
        assert spec.name == "LOTTO_6AUS49"
        assert spec.main_pick == 6
        assert spec.main_max == 49
        assert spec.sec_pick == 1
        assert spec.sec_max == 10
        assert spec.cols == ["n1","n2","n3","n4","n5","n6","super"]
    
    def test_austrian_lotto_spec(self):
        """Test Austrian Lotto specification."""
        spec = GAMES["AUSTRIAN_LOTTO"]
        assert spec.name == "AUSTRIAN_LOTTO"
        assert spec.main_pick == 6
        assert spec.main_max == 45
        assert spec.sec_pick == 0
        assert spec.sec_max == 0
        assert spec.cols == ["n1","n2","n3","n4","n5","n6"]
    
    def test_swiss_lotto_spec(self):
        """Test Swiss Lotto specification."""
        spec = GAMES["SWISS_LOTTO"]
        assert spec.name == "SWISS_LOTTO"
        assert spec.main_pick == 6
        assert spec.main_max == 42
        assert spec.sec_pick == 1
        assert spec.sec_max == 6
        assert spec.cols == ["n1","n2","n3","n4","n5","n6","lucky"]


class TestLotteryPricing:
    """Test that all lotteries have price defaults."""
    
    def test_all_lotteries_have_prices(self):
        """Test that all lotteries have ticket price defaults."""
        for lottery_name in GAMES.keys():
            assert lottery_name in OPAP_TICKET_PRICE_DEFAULTS, \
                f"{lottery_name} missing from OPAP_TICKET_PRICE_DEFAULTS"
            price = OPAP_TICKET_PRICE_DEFAULTS[lottery_name]
            assert isinstance(price, (int, float))
            assert price > 0
    
    def test_reasonable_prices(self):
        """Test that prices are in reasonable range (0.50 to 5.00 EUR)."""
        for lottery_name, price in OPAP_TICKET_PRICE_DEFAULTS.items():
            assert 0.50 <= price <= 5.00, \
                f"{lottery_name} price {price} outside reasonable range"


class TestLotteryMetadata:
    """Test lottery metadata for flags, icons, and country information."""
    
    def test_all_lotteries_have_metadata(self):
        """Test that all lotteries have metadata."""
        for lottery_name in GAMES.keys():
            assert lottery_name in LOTTERY_METADATA, \
                f"{lottery_name} missing from LOTTERY_METADATA"
    
    def test_metadata_structure(self):
        """Test that metadata has required fields."""
        required_fields = ["country", "flag", "icon", "display_name", "description", "official_url"]
        for lottery_name, metadata in LOTTERY_METADATA.items():
            for field in required_fields:
                assert field in metadata, \
                    f"{lottery_name} metadata missing field: {field}"
                assert isinstance(metadata[field], str)
                assert len(metadata[field]) > 0
    
    def test_flag_filenames(self):
        """Test that flag filenames are properly formatted."""
        for lottery_name, metadata in LOTTERY_METADATA.items():
            flag = metadata["flag"]
            assert flag.endswith(".png"), \
                f"{lottery_name} flag should be PNG format: {flag}"
    
    def test_icon_filenames(self):
        """Test that icon filenames are properly formatted."""
        for lottery_name, metadata in LOTTERY_METADATA.items():
            icon = metadata["icon"]
            assert icon.endswith(".png"), \
                f"{lottery_name} icon should be PNG format: {icon}"
    
    def test_country_assignments(self):
        """Test that countries are properly assigned."""
        expected_countries = {
            "TZOKER": "Greece",
            "LOTTO": "Greece",
            "EUROJACKPOT": "European Union",
            "UK_NATIONAL_LOTTERY": "United Kingdom",
            "LA_PRIMITIVA": "Spain",
            "SUPERENALOTTO": "Italy",
            "LOTO_FRANCE": "France",
            "LOTTO_6AUS49": "Germany",
            "AUSTRIAN_LOTTO": "Austria",
            "SWISS_LOTTO": "Switzerland",
        }
        for lottery_name, expected_country in expected_countries.items():
            assert LOTTERY_METADATA[lottery_name]["country"] == expected_country


class TestGameSpecConsistency:
    """Test consistency across game specifications."""
    
    def test_column_count_matches_picks(self):
        """Test that column count matches main_pick + sec_pick."""
        for lottery_name, spec in GAMES.items():
            expected_cols = spec.main_pick + spec.sec_pick
            actual_cols = len(spec.cols)
            assert actual_cols == expected_cols, \
                f"{lottery_name}: expected {expected_cols} columns, got {actual_cols}"
    
    def test_main_numbers_range_valid(self):
        """Test that main number range is valid."""
        for lottery_name, spec in GAMES.items():
            assert spec.main_pick > 0, f"{lottery_name}: main_pick must be positive"
            assert spec.main_max > 0, f"{lottery_name}: main_max must be positive"
            assert spec.main_pick <= spec.main_max, \
                f"{lottery_name}: can't pick {spec.main_pick} from {spec.main_max}"
    
    def test_secondary_numbers_valid(self):
        """Test that secondary number specifications are valid."""
        for lottery_name, spec in GAMES.items():
            if spec.sec_pick > 0:
                assert spec.sec_max > 0, \
                    f"{lottery_name}: sec_max must be positive when sec_pick > 0"
                assert spec.sec_pick <= spec.sec_max, \
                    f"{lottery_name}: can't pick {spec.sec_pick} from {spec.sec_max}"
            else:
                # If no secondary numbers, sec_max should be 0
                assert spec.sec_max == 0, \
                    f"{lottery_name}: sec_max should be 0 when sec_pick is 0"
