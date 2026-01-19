"""
Unit tests for non-European lottery integrations.
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ultra_lottery_helper import (
    GAMES, LOTTERY_METADATA, OPAP_TICKET_PRICE_DEFAULTS
)


class TestNonEuropeanLotterySpecs:
    """Test non-European lottery game specifications."""
    
    def test_us_powerball_spec(self):
        """Test US Powerball game specification."""
        spec = GAMES["US_POWERBALL"]
        assert spec.name == "US_POWERBALL"
        assert spec.main_pick == 5
        assert spec.main_max == 69
        assert spec.sec_pick == 1
        assert spec.sec_max == 26
        assert len(spec.cols) == 6
        assert "powerball" in spec.cols
    
    def test_us_mega_millions_spec(self):
        """Test US Mega Millions game specification."""
        spec = GAMES["US_MEGA_MILLIONS"]
        assert spec.name == "US_MEGA_MILLIONS"
        assert spec.main_pick == 5
        assert spec.main_max == 70
        assert spec.sec_pick == 1
        assert spec.sec_max == 25
        assert len(spec.cols) == 6
        assert "megaball" in spec.cols
    
    def test_australia_powerball_spec(self):
        """Test Australia Powerball game specification."""
        spec = GAMES["AUSTRALIA_POWERBALL"]
        assert spec.name == "AUSTRALIA_POWERBALL"
        assert spec.main_pick == 7
        assert spec.main_max == 35
        assert spec.sec_pick == 1
        assert spec.sec_max == 20
        assert len(spec.cols) == 8
        assert "powerball" in spec.cols
    
    def test_canada_lotto_649_spec(self):
        """Test Canada Lotto 6/49 game specification."""
        spec = GAMES["CANADA_LOTTO_649"]
        assert spec.name == "CANADA_LOTTO_649"
        assert spec.main_pick == 6
        assert spec.main_max == 49
        assert spec.sec_pick == 1
        assert spec.sec_max == 10
        assert len(spec.cols) == 7
        assert "bonus" in spec.cols
    
    def test_japan_loto_6_spec(self):
        """Test Japan Loto 6 game specification."""
        spec = GAMES["JAPAN_LOTO_6"]
        assert spec.name == "JAPAN_LOTO_6"
        assert spec.main_pick == 6
        assert spec.main_max == 43
        assert spec.sec_pick == 1
        assert spec.sec_max == 43
        assert len(spec.cols) == 7
        assert "bonus" in spec.cols
    
    def test_south_africa_powerball_spec(self):
        """Test South Africa Powerball game specification."""
        spec = GAMES["SOUTH_AFRICA_POWERBALL"]
        assert spec.name == "SOUTH_AFRICA_POWERBALL"
        assert spec.main_pick == 5
        assert spec.main_max == 50
        assert spec.sec_pick == 1
        assert spec.sec_max == 20
        assert len(spec.cols) == 6
        assert "powerball" in spec.cols


class TestNonEuropeanLotteryPricing:
    """Test non-European lottery pricing."""
    
    def test_all_non_european_lotteries_have_prices(self):
        """Test that all non-European lotteries have price defaults."""
        non_european = [
            "US_POWERBALL", "US_MEGA_MILLIONS", "AUSTRALIA_POWERBALL",
            "CANADA_LOTTO_649", "JAPAN_LOTO_6", "SOUTH_AFRICA_POWERBALL"
        ]
        for lottery in non_european:
            assert lottery in OPAP_TICKET_PRICE_DEFAULTS
            assert OPAP_TICKET_PRICE_DEFAULTS[lottery] > 0
    
    def test_pricing_reasonable(self):
        """Test that pricing is in reasonable range."""
        non_european = [
            "US_POWERBALL", "US_MEGA_MILLIONS", "AUSTRALIA_POWERBALL",
            "CANADA_LOTTO_649", "JAPAN_LOTO_6", "SOUTH_AFRICA_POWERBALL"
        ]
        for lottery in non_european:
            price = OPAP_TICKET_PRICE_DEFAULTS[lottery]
            assert 0.1 <= price <= 10.0  # Reasonable range


class TestNonEuropeanLotteryMetadata:
    """Test non-European lottery metadata."""
    
    def test_all_have_metadata(self):
        """Test that all non-European lotteries have metadata."""
        non_european = [
            "US_POWERBALL", "US_MEGA_MILLIONS", "AUSTRALIA_POWERBALL",
            "CANADA_LOTTO_649", "JAPAN_LOTO_6", "SOUTH_AFRICA_POWERBALL"
        ]
        for lottery in non_european:
            assert lottery in LOTTERY_METADATA
    
    def test_metadata_structure(self):
        """Test that metadata has required fields."""
        non_european = [
            "US_POWERBALL", "US_MEGA_MILLIONS", "AUSTRALIA_POWERBALL",
            "CANADA_LOTTO_649", "JAPAN_LOTO_6", "SOUTH_AFRICA_POWERBALL"
        ]
        required_fields = [
            "country", "flag", "icon", "display_name", 
            "description", "official_url", "results_url", "has_jackpot"
        ]
        for lottery in non_european:
            metadata = LOTTERY_METADATA[lottery]
            for field in required_fields:
                assert field in metadata
                assert metadata[field] is not None
    
    def test_countries_correct(self):
        """Test that countries are correctly assigned."""
        assert LOTTERY_METADATA["US_POWERBALL"]["country"] == "United States"
        assert LOTTERY_METADATA["US_MEGA_MILLIONS"]["country"] == "United States"
        assert LOTTERY_METADATA["AUSTRALIA_POWERBALL"]["country"] == "Australia"
        assert LOTTERY_METADATA["CANADA_LOTTO_649"]["country"] == "Canada"
        assert LOTTERY_METADATA["JAPAN_LOTO_6"]["country"] == "Japan"
        assert LOTTERY_METADATA["SOUTH_AFRICA_POWERBALL"]["country"] == "South Africa"
    
    def test_all_have_jackpots(self):
        """Test that all non-European lotteries have jackpots."""
        non_european = [
            "US_POWERBALL", "US_MEGA_MILLIONS", "AUSTRALIA_POWERBALL",
            "CANADA_LOTTO_649", "JAPAN_LOTO_6", "SOUTH_AFRICA_POWERBALL"
        ]
        for lottery in non_european:
            assert LOTTERY_METADATA[lottery]["has_jackpot"] is True
    
    def test_results_urls_valid(self):
        """Test that results URLs are present and properly formatted."""
        non_european = [
            "US_POWERBALL", "US_MEGA_MILLIONS", "AUSTRALIA_POWERBALL",
            "CANADA_LOTTO_649", "JAPAN_LOTO_6", "SOUTH_AFRICA_POWERBALL"
        ]
        for lottery in non_european:
            url = LOTTERY_METADATA[lottery]["results_url"]
            assert url.startswith("http://") or url.startswith("https://")
            assert len(url) > 10


class TestNonEuropeanColumnCounts:
    """Test column counts for non-European lotteries."""
    
    def test_us_powerball_columns(self):
        """Test US Powerball has correct column count."""
        spec = GAMES["US_POWERBALL"]
        expected = spec.main_pick + spec.sec_pick
        assert len(spec.cols) == expected
    
    def test_us_mega_millions_columns(self):
        """Test US Mega Millions has correct column count."""
        spec = GAMES["US_MEGA_MILLIONS"]
        expected = spec.main_pick + spec.sec_pick
        assert len(spec.cols) == expected
    
    def test_australia_powerball_columns(self):
        """Test Australia Powerball has correct column count."""
        spec = GAMES["AUSTRALIA_POWERBALL"]
        expected = spec.main_pick + spec.sec_pick
        assert len(spec.cols) == expected
    
    def test_canada_lotto_649_columns(self):
        """Test Canada Lotto 6/49 has correct column count."""
        spec = GAMES["CANADA_LOTTO_649"]
        expected = spec.main_pick + spec.sec_pick
        assert len(spec.cols) == expected
    
    def test_japan_loto_6_columns(self):
        """Test Japan Loto 6 has correct column count."""
        spec = GAMES["JAPAN_LOTO_6"]
        expected = spec.main_pick + spec.sec_pick
        assert len(spec.cols) == expected
    
    def test_south_africa_powerball_columns(self):
        """Test South Africa Powerball has correct column count."""
        spec = GAMES["SOUTH_AFRICA_POWERBALL"]
        expected = spec.main_pick + spec.sec_pick
        assert len(spec.cols) == expected


class TestNonEuropeanNumberRanges:
    """Test number ranges for non-European lotteries."""
    
    def test_us_powerball_ranges(self):
        """Test US Powerball number ranges."""
        spec = GAMES["US_POWERBALL"]
        assert spec.main_max == 69  # White balls 1-69
        assert spec.sec_max == 26   # Powerball 1-26
    
    def test_us_mega_millions_ranges(self):
        """Test US Mega Millions number ranges."""
        spec = GAMES["US_MEGA_MILLIONS"]
        assert spec.main_max == 70  # White balls 1-70
        assert spec.sec_max == 25   # Mega Ball 1-25
    
    def test_australia_powerball_ranges(self):
        """Test Australia Powerball number ranges."""
        spec = GAMES["AUSTRALIA_POWERBALL"]
        assert spec.main_max == 35  # Main numbers 1-35
        assert spec.sec_max == 20   # Powerball 1-20
    
    def test_canada_lotto_649_ranges(self):
        """Test Canada Lotto 6/49 number ranges."""
        spec = GAMES["CANADA_LOTTO_649"]
        assert spec.main_max == 49  # Main numbers 1-49
        assert spec.sec_max == 10   # Bonus 1-10
    
    def test_japan_loto_6_ranges(self):
        """Test Japan Loto 6 number ranges."""
        spec = GAMES["JAPAN_LOTO_6"]
        assert spec.main_max == 43  # Main numbers 1-43
        assert spec.sec_max == 43   # Bonus from same pool 1-43
    
    def test_south_africa_powerball_ranges(self):
        """Test South Africa Powerball number ranges."""
        spec = GAMES["SOUTH_AFRICA_POWERBALL"]
        assert spec.main_max == 50  # Main numbers 1-50
        assert spec.sec_max == 20   # Powerball 1-20


class TestTotalLotteryCount:
    """Test total lottery count after adding non-European lotteries."""
    
    def test_total_count(self):
        """Test that we now have 16 lotteries total."""
        assert len(GAMES) == 16
        assert len(LOTTERY_METADATA) == 16
    
    def test_all_games_have_metadata(self):
        """Test that all games have metadata."""
        for game in GAMES.keys():
            assert game in LOTTERY_METADATA
    
    def test_all_games_have_pricing(self):
        """Test that all games have pricing."""
        for game in GAMES.keys():
            assert game in OPAP_TICKET_PRICE_DEFAULTS
