# Lottery History Data

This directory contains historical draw data for all supported lotteries.

## Directory Structure

Each lottery has its own subdirectory:

```
data/history/
β"œβ"€β"€ tzoker/               # Greek TZOKER
β"œβ"€β"€ lotto/                # Greek LOTTO
β"œβ"€β"€ eurojackpot/          # EuroJackpot
β"œβ"€β"€ uk_national_lottery/  # UK National Lottery
β"œβ"€β"€ la_primitiva/         # Spanish La Primitiva
β"œβ"€β"€ superenalotto/        # Italian SuperEnalotto
β"œβ"€β"€ loto_france/          # French Loto
β"œβ"€β"€ lotto_6aus49/         # German Lotto 6aus49
β"œβ"€β"€ austrian_lotto/       # Austrian Lotto
└── swiss_lotto/          # Swiss Lotto
```

## File Format

Place historical draw data in CSV, XLS, or XLSX format in the appropriate directory.

### Required Columns

Each lottery requires specific columns based on its game specification:

#### Greek TZOKER
- `n1`, `n2`, `n3`, `n4`, `n5` - Main numbers (1-45)
- `joker` - Joker number (1-20)
- `date` - Draw date (optional)

#### Greek LOTTO
- `n1`, `n2`, `n3`, `n4`, `n5`, `n6` - Main numbers (1-49)
- `date` - Draw date (optional)

#### EuroJackpot
- `n1`, `n2`, `n3`, `n4`, `n5` - Main numbers (1-50)
- `e1`, `e2` - Euro numbers (1-12)
- `date` - Draw date (optional)

#### UK National Lottery
- `n1`, `n2`, `n3`, `n4`, `n5`, `n6` - Main numbers (1-59)
- `date` - Draw date (optional)

#### La Primitiva (Spain)
- `n1`, `n2`, `n3`, `n4`, `n5`, `n6` - Main numbers (1-49)
- `bonus` - Bonus number (0-9)
- `date` - Draw date (optional)

#### SuperEnalotto (Italy)
- `n1`, `n2`, `n3`, `n4`, `n5`, `n6` - Main numbers (1-90)
- `date` - Draw date (optional)

#### Loto France
- `n1`, `n2`, `n3`, `n4`, `n5` - Main numbers (1-49)
- `chance` - Chance number (1-10)
- `date` - Draw date (optional)

#### Lotto 6aus49 (Germany)
- `n1`, `n2`, `n3`, `n4`, `n5`, `n6` - Main numbers (1-49)
- `super` - Superzahl (0-9)
- `date` - Draw date (optional)

#### Austrian Lotto
- `n1`, `n2`, `n3`, `n4`, `n5`, `n6` - Main numbers (1-45)
- `date` - Draw date (optional)

#### Swiss Lotto
- `n1`, `n2`, `n3`, `n4`, `n5`, `n6` - Main numbers (1-42)
- `lucky` - Lucky number (1-6)
- `date` - Draw date (optional)

## Data Sources

Historical data can be obtained from:

1. **Official lottery websites** - Most provide CSV downloads
2. **Third-party aggregators** - Various sites compile lottery results
3. **Manual entry** - Create CSV files with historical results

### Example CSV Format

```csv
date,n1,n2,n3,n4,n5,n6
2024-01-01,5,12,23,34,41,49
2024-01-04,3,15,27,38,42,55
2024-01-08,8,19,25,31,44,52
```

## Automatic Merging

The application automatically merges all CSV/XLS/XLSX files found in each lottery's directory:
- Files are sorted by date (if date column exists)
- Duplicates are removed based on draw numbers
- Most recent data takes precedence

## Online Fetching

If enabled, the application can attempt to fetch recent results from official lottery websites. This is supplementary to offline data and requires internet connection.

## Privacy & Offline Use

This application is designed to work 100% offline. Online fetching is optional and disabled by default. Your lottery data stays on your computer.
