# Official Draw Results Sources

This document provides information on where to find official draw results for each supported lottery.

## European Lotteries Results Sources

### Greek Lotteries

#### TZOKER πŸ‡¬πŸ‡·
- **Official Results**: https://www.opap.gr/en/web/opap-gr/tzoker-draw-results
- **Operator**: OPAP
- **Has Jackpot**: Yes
- **Draw Frequency**: Twice weekly (Tuesday, Thursday)
- **Result Format**: 5 main numbers + 1 Joker number

#### Greek LOTTO πŸ‡¬πŸ‡·
- **Official Results**: https://www.opap.gr/en/web/opap-gr/lotto-draw-results
- **Operator**: OPAP
- **Has Jackpot**: Yes
- **Draw Frequency**: Twice weekly (Wednesday, Saturday)
- **Result Format**: 6 main numbers

### Pan-European

#### EuroJackpot πŸ‡ͺπŸ‡Ύ
- **Official Results**: https://www.eurojackpot.org/en/results/
- **Operator**: EuroJackpot consortium
- **Has Jackpot**: Yes (can reach over €100 million)
- **Draw Frequency**: Weekly (Tuesday, Friday)
- **Result Format**: 5 main numbers + 2 Euro numbers
- **Participating Countries**: 18 European countries

### United Kingdom

#### UK National Lottery πŸ‡¬πŸ‡§
- **Official Results**: https://www.national-lottery.co.uk/results/lotto/draw-history
- **Operator**: Camelot Group (National Lottery)
- **Has Jackpot**: Yes
- **Draw Frequency**: Twice weekly (Wednesday, Saturday)
- **Result Format**: 6 main numbers from 1-59
- **Additional Info**: Also offers bonus ball (not included in main game)

### Spain

#### La Primitiva πŸ‡ͺπŸ‡Έ
- **Official Results**: https://www.loteriasyapuestas.es/en/la-primitiva/results
- **Operator**: LoterΓ­as y Apuestas del Estado
- **Has Jackpot**: Yes
- **Draw Frequency**: Twice weekly (Thursday, Saturday)
- **Result Format**: 6 main numbers + 1 reintegro (refund number)

### Italy

#### SuperEnalotto πŸ‡?πŸ‡Ή
- **Official Results**: https://www.superenalotto.com/en/results
- **Operator**: Sisal
- **Has Jackpot**: Yes (known for record-breaking jackpots)
- **Draw Frequency**: Three times weekly (Tuesday, Thursday, Saturday)
- **Result Format**: 6 numbers from 1-90
- **Additional Info**: Numbers drawn from 90 (largest pool among major lotteries)

### France

#### Loto France πŸ‡«πŸ‡·
- **Official Results**: https://www.fdj.fr/jeux-de-tirage/loto/resultats
- **Operator**: FranΓ§aise des Jeux (FDJ)
- **Has Jackpot**: Yes
- **Draw Frequency**: Three times weekly (Monday, Wednesday, Saturday)
- **Result Format**: 5 main numbers + 1 Chance number (1-10)

### Germany

#### Lotto 6aus49 πŸ‡©πŸ‡ͺ
- **Official Results**: https://www.lotto.de/lotto-6aus49/lottozahlen
- **Operator**: Deutscher Lotto- und Totoblock
- **Has Jackpot**: Yes
- **Draw Frequency**: Twice weekly (Wednesday, Saturday)
- **Result Format**: 6 numbers from 1-49 + 1 Superzahl (0-9)
- **Additional Info**: Germany's most popular lottery game

### Austria

#### Austrian Lotto πŸ‡¦πŸ‡Ή
- **Official Results**: https://www.win2day.at/lottery/lotto/results
- **Operator**: Γ–sterreichische Lotterien
- **Has Jackpot**: Yes
- **Draw Frequency**: Twice weekly (Wednesday, Sunday)
- **Result Format**: 6 numbers from 1-45

### Switzerland

#### Swiss Lotto πŸ‡¨πŸ‡­
- **Official Results**: https://www.swisslos.ch/en/swisslotto/information/winning-numbers.html
- **Operator**: Swisslos
- **Has Jackpot**: Yes
- **Draw Frequency**: Twice weekly (Wednesday, Saturday)
- **Result Format**: 6 numbers from 1-42 + 1 Lucky number (1-6)

## How to Use Results Data

### Downloading Historical Data

Most lottery operators provide:
1. **Recent Results**: Last 30-180 days available online
2. **Historical Archives**: Some offer CSV/Excel downloads
3. **APIs**: Limited availability, mostly unofficial

### Data Collection Tips

1. **Manual Download**: Visit results URL and download available data
2. **Web Scraping**: Be respectful of rate limits and terms of service
3. **Third-party Sources**: Various lottery aggregation sites exist
4. **Official Apps**: Some lotteries offer mobile apps with result history

### Importing into Ultra Lottery Helper

Place downloaded results in CSV format in:
```
data/history/{lottery_name}/
```

Example:
```
data/history/uk_national_lottery/results_2024.csv
data/history/superenalotto/draws_2023.csv
```

The application will automatically merge all files found in each lottery's directory.

## Jackpot Information

All listed lotteries offer **progressive jackpots** that:
- Start at a minimum guaranteed amount
- Roll over when not won
- Can reach significant amounts (especially EuroJackpot and SuperEnalotto)
- Are subject to taxation based on local laws

### Jackpot Size Ranges

Typical jackpot ranges (approximate):
- **EuroJackpot**: €10 million - €120 million
- **SuperEnalotto**: €2 million - €200+ million (record: €371 million)
- **UK National Lottery**: £2 million - £20+ million
- **German Lotto**: €1 million - €45 million
- **French Loto**: €2 million - €20+ million
- **La Primitiva**: €3 million - €30+ million
- **Austrian Lotto**: €1 million - €10+ million
- **Swiss Lotto**: CHF 1 million - CHF 50+ million
- **TZOKER**: €200,000 - €5+ million
- **Greek LOTTO**: €300,000 - €10+ million

## Legal Disclaimer

This information is provided for reference purposes only. Always verify results on official lottery websites. The Ultra Lottery Helper application is for entertainment and analytical purposes. Play responsibly and within your means.

Lottery results and jackpot amounts are subject to change. Prize amounts may vary based on currency exchange rates, local taxes, and lottery rules.
