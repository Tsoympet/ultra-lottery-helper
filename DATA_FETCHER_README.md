# Lottery Data Fetcher - Live Feed Module

Automated data fetching system for lottery draw results with local storage and update tracking.

## Features

- **Automated Fetching**: Fetch latest draw results from official lottery sources
- **Smart Updates**: Tracks last fetch time to avoid excessive requests (6-hour minimum interval)
- **Local Storage**: Saves fetched data as timestamped CSV files
- **Fetch Logging**: JSON log tracking when each lottery was last updated
- **Status Monitoring**: View fetch status for all lotteries
- **Batch Operations**: Fetch all lotteries or specific ones
- **Respectful Scraping**: Built-in delays between requests

## Quick Start

### Command Line Usage

```bash
# Show fetch status for all lotteries
python src/lottery_data_fetcher.py --status

# Fetch a specific lottery
python src/lottery_data_fetcher.py --game TZOKER

# Fetch all lotteries
python src/lottery_data_fetcher.py --all

# Force fetch even if recently updated
python src/lottery_data_fetcher.py --game UK_NATIONAL_LOTTERY --force
```

### Python API Usage

```python
from lottery_data_fetcher import LotteryDataFetcher, fetch_latest_draws

# Simple way - fetch a specific lottery
results = fetch_latest_draws(game='EUROJACKPOT')
print(results)

# Fetch all lotteries
results = fetch_latest_draws()

# Advanced usage with fetcher object
fetcher = LotteryDataFetcher()

# Fetch specific lottery
success, msg = fetcher.fetch_lottery_data('TZOKER')
print(msg)

# Get status of all fetches
status_df = fetcher.get_fetch_status()
print(status_df)

# Fetch all with force update
results = fetcher.fetch_all_lotteries(force=True)
```

## How It Works

### Data Flow

1. **Check Last Fetch**: Consults `fetch_log.json` to avoid duplicate requests
2. **Fetch Online**: Uses `fetch_online_history()` from core module
3. **Save Data**: Stores results as `fetched_YYYYMMDD_HHMMSS.csv` in lottery's data directory
4. **Update Log**: Records fetch time, row count, and status
5. **Merge**: Core module automatically merges all CSV files during prediction

### Directory Structure

```
data/
β"œβ"€β"€ history/
β"‚   β"œβ"€β"€ tzoker/
β"‚   β"‚   β"œβ"€β"€ fetched_20260119_120000.csv
β"‚   β"‚   └── fetched_20260119_180000.csv
β"‚   β"œβ"€β"€ uk_national_lottery/
β"‚   β"‚   └── fetched_20260119_120500.csv
β"‚   └── ...
└── fetch_log.json
```

### Fetch Log Format

```json
{
  "TZOKER": {
    "last_fetch": "2026-01-19T12:00:00",
    "rows_fetched": 150,
    "file": "data/history/tzoker/fetched_20260119_120000.csv",
    "status": "success"
  },
  "UK_NATIONAL_LOTTERY": {
    "last_fetch": "2026-01-19T12:05:00",
    "rows_fetched": 200,
    "file": "data/history/uk_national_lottery/fetched_20260119_120500.csv",
    "status": "success"
  }
}
```

## Supported Lotteries

All 10 lotteries are supported:

| Lottery | Code | Country |
|---------|------|---------|
| TZOKER | `TZOKER` | πŸ‡¬πŸ‡· Greece |
| Greek LOTTO | `LOTTO` | πŸ‡¬πŸ‡· Greece |
| EuroJackpot | `EUROJACKPOT` | πŸ‡ͺπŸ‡Ί Pan-European |
| UK National Lottery | `UK_NATIONAL_LOTTERY` | πŸ‡¬πŸ‡§ United Kingdom |
| La Primitiva | `LA_PRIMITIVA` | πŸ‡ͺπŸ‡Έ Spain |
| SuperEnalotto | `SUPERENALOTTO` | πŸ‡?πŸ‡Ή Italy |
| Loto France | `LOTO_FRANCE` | πŸ‡«πŸ‡· France |
| Lotto 6aus49 | `LOTTO_6AUS49` | πŸ‡©πŸ‡ͺ Germany |
| Austrian Lotto | `AUSTRIAN_LOTTO` | πŸ‡¦πŸ‡Ή Austria |
| Swiss Lotto | `SWISS_LOTTO` | πŸ‡¨πŸ‡­ Switzerland |

## Configuration

### Update Frequency

Default: 6 hours minimum between fetches

To change, modify the check in `fetch_lottery_data()`:

```python
# Don't fetch if updated in last N hours
if hours_since < 6:  # Change this value
    return False, f"..."
```

### Request Delays

Default: 2 seconds between lottery requests

To change, modify the delay in `fetch_all_lotteries()`:

```python
if success:
    time.sleep(2)  # Change this value
```

## Integration with Prediction System

The fetched data is automatically used by the prediction system:

1. Fetched CSV files are placed in `data/history/{lottery}/`
2. When generating predictions, the core module calls `_load_all_history()`
3. This function automatically merges ALL CSV files in the lottery's directory
4. The ML models train on the complete merged dataset

No additional integration needed - it just works!

## Scheduling Automated Updates

### Option 1: Cron (Linux/Mac)

```bash
# Add to crontab (crontab -e)
# Fetch all lotteries twice daily at 2 AM and 2 PM
0 2,14 * * * cd /path/to/ultra-lottery-helper && python src/lottery_data_fetcher.py --all

# Fetch specific lottery every 6 hours
0 */6 * * * cd /path/to/ultra-lottery-helper && python src/lottery_data_fetcher.py --game EUROJACKPOT
```

### Option 2: Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Trigger: Daily at specific time
4. Action: Start a program
5. Program: `python`
6. Arguments: `src/lottery_data_fetcher.py --all`
7. Start in: `C:\path\to\ultra-lottery-helper`

### Option 3: Python APScheduler (Advanced)

For more control, use APScheduler:

```python
from apscheduler.schedulers.blocking import BlockingScheduler
from lottery_data_fetcher import LotteryDataFetcher

scheduler = BlockingScheduler()
fetcher = LotteryDataFetcher()

# Fetch all lotteries every 12 hours
@scheduler.scheduled_job('interval', hours=12)
def fetch_all_job():
    print("Running scheduled fetch...")
    fetcher.fetch_all_lotteries()

scheduler.start()
```

## Error Handling

The fetcher handles errors gracefully:

- **Network errors**: Logged and skipped, doesn't crash
- **Parse errors**: Returns empty DataFrame with error message
- **Rate limiting**: 6-hour minimum interval prevents excessive requests
- **Missing data**: Logged in fetch log with 'failed' status

## Best Practices

1. **Don't abuse sources**: Respect rate limits and use reasonable intervals
2. **Monitor fetch log**: Check `fetch_log.json` for failed fetches
3. **Verify data**: Spot-check fetched CSV files for accuracy
4. **Backup**: Keep backups of important historical data
5. **Legal compliance**: Ensure data usage complies with lottery operator terms

## Troubleshooting

### "No HTML tables found"

The lottery website structure may have changed. Check:
1. Visit the results URL in a browser
2. Verify tables are present
3. Update scraping logic in `fetch_online_history()` if needed

### "Last fetched X hours ago"

This is normal - prevents excessive requests. Use `--force` to override:

```bash
python src/lottery_data_fetcher.py --game TZOKER --force
```

### Network timeouts

Increase timeout in `fetch_online_history()`:

```python
response = requests.get(urls[game], timeout=30)  # Increase from 10
```

## Future Enhancements

Planned features:
- API integration where available (instead of scraping)
- Email notifications on fetch failures
- Webhook support for real-time updates
- Retry logic with exponential backoff
- Multi-threaded fetching for faster batch operations
- Data validation and anomaly detection

## License

MIT License - Same as main project

## Contributing

To add support for a new lottery:

1. Add lottery spec to `GAMES` in `ultra_lottery_helper.py`
2. Add metadata to `LOTTERY_METADATA` with `results_url`
3. Update `fetch_online_history()` URL mapping
4. Test fetching with `--game YOUR_LOTTERY`
5. Submit PR with documentation

## Support

For issues or questions:
1. Check fetch log: `cat data/fetch_log.json`
2. Run with `--status` to see current state
3. Try `--force` to bypass cache
4. Review error messages in console output
