# Lottery Data Scheduler - Automated Scheduling System

Complete scheduling system for automated lottery data fetching with configurable intervals, retry logic, and multiple backend options.

## Features

- **Automated Scheduling**: Set-and-forget data fetching on configurable intervals
- **Two Backends**: 
  - Simple interval-based (no dependencies)
  - APScheduler-based (advanced features, optional)
- **Per-Lottery Configuration**: Different schedules for different lotteries
- **Retry Logic**: Automatic retries with exponential backoff
- **Cron Support**: Advanced time-based scheduling (with APScheduler)
- **Status Monitoring**: View current schedule configuration
- **Persistent Config**: JSON-based configuration storage

## Quick Start

### 1. Enable All Lotteries

```bash
# Enable automatic fetching for all lotteries (every 12 hours)
python src/lottery_scheduler.py --enable-all

# Custom interval (every 6 hours)
python src/lottery_scheduler.py --enable-all --interval 6
```

### 2. Start the Scheduler

```bash
# Start with simple backend (no extra dependencies)
python src/lottery_scheduler.py --start

# Start with APScheduler backend (more features)
python src/lottery_scheduler.py --start --use-apscheduler
```

### 3. Check Status

```bash
python src/lottery_scheduler.py --status
```

## Installation

### Basic (Simple Scheduler)

No additional dependencies needed - works with existing requirements:

```bash
pip install -r requirements.txt
```

### Advanced (APScheduler)

For cron support and advanced features:

```bash
pip install apscheduler
```

## Usage

### Command-Line Interface

```bash
# Enable all lotteries with 12-hour interval
python src/lottery_scheduler.py --enable-all

# Add specific lottery to schedule
python src/lottery_scheduler.py --add EUROJACKPOT --interval 8

# Remove lottery from schedule
python src/lottery_scheduler.py --remove TZOKER

# View scheduler status
python src/lottery_scheduler.py --status

# Start scheduler (runs in foreground)
python src/lottery_scheduler.py --start

# Start with APScheduler backend
python src/lottery_scheduler.py --start --use-apscheduler
```

### Python API

```python
from lottery_scheduler import LotteryScheduler

# Create scheduler
scheduler = LotteryScheduler()

# Enable all lotteries
scheduler.enable_all_lotteries(interval_hours=12)

# Add specific lottery
scheduler.add_schedule('EUROJACKPOT', interval_hours=8)

# Add with cron expression (requires APScheduler)
scheduler = LotteryScheduler(use_apscheduler=True)
scheduler.add_schedule('TZOKER', cron_expression='0 */6 * * *')

# Start scheduler (APScheduler)
scheduler.start_apscheduler()

# Or start simple scheduler (blocks)
scheduler.start_simple_scheduler()

# Get status
status = scheduler.get_status()
print(status)

# Stop scheduler
scheduler.stop()
```

## Configuration

Configuration is stored in `data/schedule_config.json`:

```json
{
  "enabled": true,
  "default_interval_hours": 12,
  "schedules": {
    "TZOKER": {
      "enabled": true,
      "interval_hours": 12
    },
    "EUROJACKPOT": {
      "enabled": true,
      "interval_hours": 12,
      "cron": "0 */12 * * *"
    }
  },
  "global_settings": {
    "respect_cache": true,
    "batch_delay_seconds": 2,
    "max_retries": 3,
    "retry_delay_minutes": 5
  }
}
```

### Configuration Options

#### Global Settings

- **respect_cache**: Honor 6-hour minimum cache interval (default: true)
- **batch_delay_seconds**: Delay between lottery fetches in batch (default: 2)
- **max_retries**: Maximum retry attempts on failure (default: 3)
- **retry_delay_minutes**: Minutes between retries (default: 5)

#### Per-Lottery Settings

- **enabled**: Whether to fetch this lottery (default: true)
- **interval_hours**: Hours between fetches (default: 12)
- **cron**: Cron expression (APScheduler only, optional)

## Scheduling Backends

### Simple Backend (Default)

**Pros:**
- No extra dependencies
- Easy to understand
- Lightweight

**Cons:**
- Only interval-based scheduling
- Must run in foreground
- Less precise timing

**When to use:** For simple setups, Docker containers, or when you don't need cron.

### APScheduler Backend

**Pros:**
- Cron expression support
- Background execution
- More precise timing
- Job persistence

**Cons:**
- Requires extra dependency
- Slightly more complex

**When to use:** For production deployments, complex schedules, or when using cron.

## Scheduling Strategies

### Strategy 1: Equal Intervals for All

Best for: Simple setups, similar draw frequencies

```python
scheduler = LotteryScheduler()
scheduler.enable_all_lotteries(interval_hours=12)
scheduler.start_simple_scheduler()
```

### Strategy 2: Custom Per-Lottery

Best for: Different draw frequencies, optimizing traffic

```python
scheduler = LotteryScheduler()

# High-frequency lotteries (3x weekly)
scheduler.add_schedule('LOTO_FRANCE', interval_hours=8)
scheduler.add_schedule('SUPERENALOTTO', interval_hours=8)

# Medium-frequency lotteries (2x weekly)
scheduler.add_schedule('TZOKER', interval_hours=12)
scheduler.add_schedule('EUROJACKPOT', interval_hours=12)

# Low-frequency lotteries (2x weekly but less critical)
scheduler.add_schedule('AUSTRIAN_LOTTO', interval_hours=24)

scheduler.start_simple_scheduler()
```

### Strategy 3: Cron-Based (APScheduler)

Best for: Precise timing, matching draw schedules

```python
scheduler = LotteryScheduler(use_apscheduler=True)

# Fetch TZOKER at 9 PM on draw days (Tue, Thu)
scheduler.add_schedule('TZOKER', cron_expression='0 21 * * 2,4')

# Fetch EuroJackpot at 10 PM on draw days (Tue, Fri)
scheduler.add_schedule('EUROJACKPOT', cron_expression='0 22 * * 2,5')

# Fetch UK Lottery at 10 PM on draw days (Wed, Sat)
scheduler.add_schedule('UK_NATIONAL_LOTTERY', cron_expression='0 22 * * 3,6')

scheduler.start_apscheduler()
```

## Running as a Service

### Linux (systemd)

Create `/etc/systemd/system/lottery-scheduler.service`:

```ini
[Unit]
Description=Lottery Data Scheduler
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/ultra-lottery-helper
ExecStart=/usr/bin/python3 src/lottery_scheduler.py --start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable lottery-scheduler
sudo systemctl start lottery-scheduler
sudo systemctl status lottery-scheduler
```

### Windows (Task Scheduler)

1. Open Task Scheduler
2. Create Basic Task: "Lottery Data Scheduler"
3. Trigger: At startup
4. Action: Start a program
   - Program: `pythonw.exe` (for background)
   - Arguments: `src\lottery_scheduler.py --start --use-apscheduler`
   - Start in: `C:\path\to\ultra-lottery-helper`

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt apscheduler

COPY src/ ./src/
COPY data/ ./data/

CMD ["python", "src/lottery_scheduler.py", "--start", "--use-apscheduler"]
```

Build and run:

```bash
docker build -t lottery-scheduler .
docker run -d --name lottery-scheduler \
  -v $(pwd)/data:/app/data \
  lottery-scheduler
```

## Monitoring

### View Logs

```bash
# Tail logs (when running in foreground)
python src/lottery_scheduler.py --start | tee scheduler.log

# Or redirect to file
python src/lottery_scheduler.py --start > scheduler.log 2>&1 &
```

### Check Status

```bash
# Quick status check
python src/lottery_scheduler.py --status

# Detailed fetch history
python src/lottery_data_fetcher.py --status
```

### Monitor Fetch Log

```bash
# View recent fetches
cat data/fetch_log.json | python -m json.tool

# Watch for changes (Linux/Mac)
watch -n 60 'cat data/fetch_log.json | python -m json.tool'
```

## Troubleshooting

### Scheduler Won't Start

**Problem:** "Scheduler is not enabled!"

**Solution:**
```bash
python src/lottery_scheduler.py --enable-all
```

### No Data Being Fetched

**Problem:** Scheduler running but no data

**Solutions:**
1. Check logs for errors
2. Verify network connectivity
3. Test manual fetch: `python src/lottery_data_fetcher.py --game TZOKER --force`
4. Check cache intervals in config

### APScheduler Not Working

**Problem:** Can't use cron or background mode

**Solution:**
```bash
pip install apscheduler
python src/lottery_scheduler.py --start --use-apscheduler
```

### High Memory Usage

**Problem:** Scheduler using too much memory

**Solutions:**
1. Use simple backend instead of APScheduler
2. Increase interval times
3. Reduce number of scheduled lotteries
4. Run as separate processes per lottery

## Best Practices

1. **Start Small**: Begin with longer intervals (24h) and adjust down
2. **Respect Sources**: Don't fetch more than 2-3 times daily per lottery
3. **Monitor First**: Run for a week and check logs before deploying
4. **Use Caching**: Keep `respect_cache: true` to honor minimum intervals
5. **Stagger Schedules**: Avoid fetching all lotteries simultaneously
6. **Log Everything**: Keep logs for debugging and compliance
7. **Test Manually**: Verify fetching works before scheduling

## Integration with Prediction System

The scheduler automatically integrates with the prediction system:

1. **Scheduler** runs at configured intervals
2. **Fetcher** downloads latest results
3. **CSV files** saved to `data/history/{lottery}/`
4. **Prediction system** auto-merges all CSVs when generating predictions
5. **ML models** train on updated data

No manual intervention needed!

## Advanced Usage

### Custom Retry Logic

```python
scheduler = LotteryScheduler()

# Modify retry settings
scheduler.schedule_config['global_settings']['max_retries'] = 5
scheduler.schedule_config['global_settings']['retry_delay_minutes'] = 10
scheduler._save_schedule_config()
```

### Email Notifications (Example)

```python
import smtplib
from email.message import EmailMessage

class NotifyingScheduler(LotteryScheduler):
    def _fetch_with_retry(self, game):
        success = super()._fetch_with_retry(game)
        
        if not success:
            # Send email notification
            msg = EmailMessage()
            msg['Subject'] = f'Lottery fetch failed: {game}'
            msg['From'] = 'scheduler@example.com'
            msg['To'] = 'admin@example.com'
            msg.set_content(f'Failed to fetch {game} after retries')
            
            # Send email (configure SMTP)
            # smtp.send_message(msg)
        
        return success
```

### Webhook Notifications

```python
import requests

class WebhookScheduler(LotteryScheduler):
    def _run_scheduled_fetch(self, game):
        success = super()._run_scheduled_fetch(game)
        
        # Send webhook
        requests.post('https://your-webhook.com/lottery-update', json={
            'lottery': game,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
```

## Performance Considerations

- **Memory**: Simple backend uses ~10-20MB, APScheduler ~30-50MB
- **CPU**: Minimal during idle, spikes during fetch
- **Network**: Depends on lottery site response size (typically <1MB per fetch)
- **Disk**: ~100-500KB per lottery fetch (CSV files)

## Security

- **No Credentials**: Scheduler doesn't store passwords
- **Local Only**: Data stays on your machine
- **HTTPS**: All lottery fetches use HTTPS
- **Rate Limiting**: Built-in delays respect server limits
- **Error Handling**: Fails gracefully without crashing

## License

MIT License - Same as main project

## Support

For issues:
1. Check `data/schedule_config.json` for configuration
2. Review logs for error messages
3. Test manual fetch first
4. Verify APScheduler installed (if using)

## Future Enhancements

Planned features:
- Web UI for configuration
- Real-time status dashboard
- Email/SMS notifications
- Metric collection and analytics
- Cloud deployment templates
- Auto-scaling based on load
