# Lottery Assets Guide

This directory contains flags and lottery icons for all supported European lotteries.

## Directory Structure

```
assets/
β"œβ"€β"€ flags/           # Country flags for each lottery
β"œβ"€β"€ lottery_icons/   # Official lottery icons
β"œβ"€β"€ icon.ico         # Main application icon
β"œβ"€β"€ banner.bmp       # Installer banner
β"œβ"€β"€ banner_small.bmp # Small installer banner
└── splash.png       # Application splash screen
```

## European Lotteries Assets

### Required Flag Files (SVG or PNG format, 64x64 recommended)

Place country flag images in `assets/flags/`:

1. **uk.png** - United Kingdom flag (for UK National Lottery)
2. **spain.png** - Spain flag (for La Primitiva)
3. **italy.png** - Italy flag (for SuperEnalotto)
4. **france.png** - France flag (for Loto France)
5. **germany.png** - Germany flag (for Lotto 6aus49)
6. **austria.png** - Austria flag (for Austrian Lotto)
7. **switzerland.png** - Switzerland flag (for Swiss Lotto)
8. **greece.png** - Greece flag (for existing TZOKER, LOTTO)
9. **european_union.png** - EU flag (for EuroJackpot)

### Required Lottery Icon Files (PNG format, 64x64 recommended)

Place official lottery icons in `assets/lottery_icons/`:

1. **uk_national_lottery.png** - UK National Lottery official logo
2. **la_primitiva.png** - La Primitiva official logo
3. **superenalotto.png** - SuperEnalotto official logo
4. **loto_france.png** - Loto France (FDJ) official logo
5. **lotto_6aus49.png** - German Lotto official logo
6. **austrian_lotto.png** - Austrian Lotto official logo
7. **swiss_lotto.png** - Swiss Lotto official logo
8. **tzoker.png** - TZOKER official logo
9. **lotto_greece.png** - Greek Lotto official logo
10. **eurojackpot.png** - EuroJackpot official logo

## Asset Specifications

### Flags
- Format: PNG or SVG
- Recommended size: 64x64 pixels
- Aspect ratio: 1:1 (square) or 3:2 (standard flag ratio)
- Background: Transparent (PNG) or white

### Lottery Icons
- Format: PNG with transparency
- Recommended size: 64x64 pixels
- Must be official logos (ensure proper licensing/trademark compliance)
- Background: Transparent

## Sources for Assets

### Country Flags
You can obtain free country flags from:
- https://flagpedia.net/ (Public domain)
- https://www.flaticon.com/packs/countrys-flags (Free with attribution)
- https://github.com/lipis/flag-icons (MIT License)

### Lottery Icons
**Important**: Official lottery logos are trademarked. For production use:
1. Contact the lottery operator for permission
2. Use only for reference/educational purposes
3. Create derivative works that don't infringe on trademarks
4. Consider using generic lottery ball/ticket icons instead

## Integration in Code

The lottery metadata is defined in `src/ultra_lottery_helper.py`:

```python
LOTTERY_METADATA = {
    "UK_NATIONAL_LOTTERY": {
        "country": "United Kingdom",
        "flag": "uk.png",
        "icon": "uk_national_lottery.png",
        "official_url": "https://www.national-lottery.co.uk",
    },
    # ... other lotteries
}
```

## Legal Notice

This software is for educational and personal use. Lottery names, logos, and trademarks are property of their respective owners. No endorsement by lottery operators is implied.

When distributing this software:
1. Ensure you have rights to use any included images
2. Use generic placeholder icons if official logos cannot be obtained
3. Include proper attribution for flag assets
4. Respect trademark and copyright laws
