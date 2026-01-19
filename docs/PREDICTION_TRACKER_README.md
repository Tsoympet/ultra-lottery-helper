# Lottery Prediction Tracker - Prediction Comparison and Validation System

Complete system for tracking lottery predictions, comparing them with actual draw results, and analyzing prediction accuracy over time.

## Features

- **Prediction Storage**: Save predictions for future draws with metadata
- **Automatic Comparison**: Compare predictions with actual draw results
- **Accuracy Tracking**: Track prediction accuracy over time
- **Statistics**: Detailed accuracy metrics and performance analysis
- **Prize Detection**: Automatically determine if predictions would win prizes
- **Historical Analysis**: Analyze prediction performance over custom time periods
- **Pending Tracking**: View all pending predictions awaiting results

## Quick Start

### Save a Prediction

```bash
# Save prediction for EuroJackpot draw on 2026-01-25
python src/prediction_tracker.py --save EUROJACKPOT \
    --numbers "5,12,18,27,33,2,8" \
    --draw-date "2026-01-25"
```

### Compare with Actual Results

```bash
# Compare predictions with actual draw results
python src/prediction_tracker.py --compare EUROJACKPOT \
    --draw-date "2026-01-25" \
    --actual "5,12,19,27,33,2,9"
```

### Auto-Compare All Pending Predictions

```bash
# Automatically compare all pending predictions with latest draws
python src/prediction_tracker.py --auto-compare
```

### View Statistics

```bash
# View accuracy statistics (last 30 days)
python src/prediction_tracker.py --stats

# View statistics for specific lottery
python src/prediction_tracker.py --stats --game TZOKER

# View statistics for last 90 days
python src/prediction_tracker.py --stats --days 90
```

### View Pending Predictions

```bash
# View all pending predictions
python src/prediction_tracker.py --pending

# View pending for specific lottery
python src/prediction_tracker.py --pending --game UK_NATIONAL_LOTTERY
```

## Python API

### Save Predictions

```python
from prediction_tracker import PredictionTracker

tracker = PredictionTracker()

# Save a prediction
pred_id = tracker.save_prediction(
    game='EUROJACKPOT',
    draw_date='2026-01-25',
    predicted_numbers=[5, 12, 18, 27, 33, 2, 8],
    metadata={
        'confidence': 0.85,
        'method': 'ML_ensemble',
        'model_version': '1.2.3'
    }
)

print(f"Prediction saved: {pred_id}")
```

### Compare with Results

```python
# Compare with actual draw
result = tracker.compare_with_draw(
    game='EUROJACKPOT',
    draw_date='2026-01-25',
    actual_numbers=[5, 12, 19, 27, 33, 2, 9]
)

print(f"Matches: {result['results'][0]['match_count']}")
print(f"Accuracy: {result['results'][0]['accuracy']}%")
print(f"Prize: {result['results'][0]['prize_tier']}")
```

### Auto-Compare

```python
# Automatically compare all pending predictions with latest draws
results = tracker.auto_compare_latest()

for game, result in results.items():
    if 'results' in result:
        print(f"{game}: {result['predictions_evaluated']} predictions evaluated")
```

### Get Statistics

```python
# Get accuracy statistics
stats_df = tracker.get_accuracy_stats(days_back=30)
print(stats_df)

# Output:
#         Lottery  Predictions  Avg Matches  Max Matches  Avg Accuracy %  Prize Wins  Win Rate %
#  EuroJackpot           15         2.13            4            30.43           3       20.00
#       TZOKER           10         1.80            3            25.71           1       10.00
```

### Get Pending Predictions

```python
# Get pending predictions
pending_df = tracker.get_pending_predictions()
print(pending_df)

# Output:
#        Lottery   Draw Date              Numbers                 Created                            ID
#  EuroJackpot  2026-01-25  5, 12, 18, 27, 33, 2, 8  2026-01-19T...  EUROJACKPOT_2026-01-25_...
```

## How It Works

### 1. Save Prediction

When you save a prediction:
1. Numbers are validated against lottery rules
2. Prediction is stored in `data/history/predictions.json`
3. Unique ID is generated: `{GAME}_{DATE}_{TIMESTAMP}`
4. Status is set to "pending"

### 2. Compare with Results

When comparing:
1. Finds all pending predictions for that draw
2. Calculates matches between predicted and actual numbers
3. Separates main and secondary number matches
4. Calculates accuracy percentage
5. Determines prize tier (simplified algorithm)
6. Updates prediction status to "matched"
7. Stores result in `data/history/prediction_results.json`

### 3. Auto-Compare

Auto-compare feature:
1. Loads latest draw from historical data
2. Finds pending predictions for that draw date
3. Automatically runs comparison
4. Updates prediction statuses

## Data Storage

### Predictions File (`predictions.json`)

```json
{
  "EUROJACKPOT": [
    {
      "id": "EUROJACKPOT_2026-01-25_20260119120000",
      "game": "EUROJACKPOT",
      "draw_date": "2026-01-25",
      "predicted_numbers": [5, 12, 18, 27, 33, 2, 8],
      "created_at": "2026-01-19T12:00:00",
      "metadata": {
        "confidence": 0.85,
        "method": "ML_ensemble"
      },
      "status": "pending"
    }
  ]
}
```

### Results File (`prediction_results.json`)

```json
{
  "EUROJACKPOT": [
    {
      "prediction_id": "EUROJACKPOT_2026-01-25_20260119120000",
      "draw_date": "2026-01-25",
      "predicted_numbers": [5, 12, 18, 27, 33, 2, 8],
      "actual_numbers": [5, 12, 19, 27, 33, 2, 9],
      "matched_numbers": [5, 12, 27, 33, 2],
      "match_count": 5,
      "main_matches": 4,
      "sec_matches": 1,
      "accuracy": 71.43,
      "prize_tier": "Tier 2 (4 matches)",
      "timestamp": "2026-01-25T21:00:00"
    }
  ]
}
```

## Integration with Scheduler

Combine with scheduler for automated tracking:

```python
from lottery_scheduler import LotteryScheduler
from prediction_tracker import PredictionTracker
from ultra_lottery_helper import generate_candidates, GAMES

class AutoPredictionScheduler(LotteryScheduler):
    def __init__(self):
        super().__init__()
        self.tracker = PredictionTracker()
    
    def _run_scheduled_fetch(self, game):
        # Fetch new data
        super()._run_scheduled_fetch(game)
        
        # Auto-compare pending predictions
        results = self.tracker.auto_compare_latest(game)
        
        # Log results
        if results:
            print(f"Auto-compared predictions for {game}")
```

## Prize Tier Detection

The system includes a simplified prize tier algorithm:

- **Jackpot**: All main + all secondary numbers match
- **Tier 2**: 5+ main number matches
- **Tier 3**: 4 main number matches
- **Tier 4**: 3 main number matches
- **No Prize**: < 3 matches

**Note**: Real prize tiers are lottery-specific and more complex. This provides a general indication.

## Statistics Explained

### Metrics

- **Predictions**: Total number of predictions evaluated
- **Avg Matches**: Average number of matching numbers
- **Max Matches**: Best prediction (most matches)
- **Avg Accuracy %**: Average percentage of correct numbers
- **Prize Wins**: Number of predictions that won prizes
- **Win Rate %**: Percentage of predictions that won prizes

### Example Output

```
=== Prediction Accuracy Statistics ===
         Lottery  Predictions  Avg Matches  Max Matches  Avg Accuracy %  Prize Wins  Win Rate %
     EuroJackpot           15         2.13            4            30.43           3       20.00
          TZOKER           10         1.80            3            25.71           1       10.00
  UK National...            8         1.75            3            29.17           1       12.50
```

## Best Practices

### 1. Save Predictions Before Draw

Save predictions before the draw occurs:

```bash
# Good: Save prediction for future draw
python src/prediction_tracker.py --save TZOKER \
    --numbers "3,12,18,24,30,15" \
    --draw-date "2026-01-22"
```

### 2. Use Metadata

Store useful metadata:

```python
tracker.save_prediction(
    game='TZOKER',
    draw_date='2026-01-22',
    predicted_numbers=[3, 12, 18, 24, 30, 15],
    metadata={
        'method': 'ML_ensemble',
        'confidence': 0.78,
        'top_features': ['frequency', 'recency'],
        'model_accuracy': 0.82
    }
)
```

### 3. Regular Auto-Compare

Run auto-compare after each draw:

```bash
# In cron or scheduler
0 22 * * 2,5 cd /path/to/app && python src/prediction_tracker.py --auto-compare
```

### 4. Monitor Statistics

Review stats weekly:

```bash
python src/prediction_tracker.py --stats --days 7
```

### 5. Clean Old Data

Periodically clean old predictions:

```python
tracker = PredictionTracker()
removed = tracker.clear_old_predictions(days_old=90)
print(f"Removed {removed} old predictions")
```

## Advanced Usage

### Batch Prediction Saving

```python
tracker = PredictionTracker()

# Save multiple predictions for different lotteries
predictions = [
    ('EUROJACKPOT', '2026-01-25', [5, 12, 18, 27, 33, 2, 8]),
    ('TZOKER', '2026-01-21', [3, 12, 18, 24, 30, 15]),
    ('UK_NATIONAL_LOTTERY', '2026-01-22', [5, 12, 18, 27, 33, 42])
]

for game, date, numbers in predictions:
    pred_id = tracker.save_prediction(game, date, numbers)
    print(f"Saved: {pred_id}")
```

### Custom Analysis

```python
tracker = PredictionTracker()

# Analyze specific lottery over time
for days in [7, 30, 90]:
    stats = tracker.get_accuracy_stats(game='EUROJACKPOT', days_back=days)
    if not stats.empty:
        avg_acc = stats.iloc[0]['Avg Accuracy %']
        print(f"Last {days} days: {avg_acc}% avg accuracy")
```

### Export Statistics

```python
import pandas as pd

tracker = PredictionTracker()

# Get statistics and export to CSV
stats_df = tracker.get_accuracy_stats(days_back=90)
stats_df.to_csv('prediction_stats.csv', index=False)

# Get detailed results
if 'EUROJACKPOT' in tracker.results:
    results_df = pd.DataFrame(tracker.results['EUROJACKPOT'])
    results_df.to_csv('eurojackpot_results.csv', index=False)
```

## Troubleshooting

### "No predictions found"

**Problem**: Auto-compare finds no predictions

**Solution**: Save predictions with correct draw dates before running auto-compare

### "No historical data available"

**Problem**: Cannot compare because no draw data exists

**Solution**: Ensure data fetcher has run and historical data exists in `data/history/{lottery}/`

### Incorrect Match Counts

**Problem**: Match counts seem wrong

**Solution**: Verify predicted numbers match lottery format (main + secondary)

### Missing Statistics

**Problem**: Stats command shows no data

**Solution**: 
1. Compare predictions first to generate results
2. Check date range (`--days` parameter)
3. Verify results file exists

## Integration Examples

### With Data Fetcher

```python
from lottery_data_fetcher import LotteryDataFetcher
from prediction_tracker import PredictionTracker

fetcher = LotteryDataFetcher()
tracker = PredictionTracker()

# Fetch latest data
fetcher.fetch_lottery_data('EUROJACKPOT')

# Auto-compare predictions
results = tracker.auto_compare_latest('EUROJACKPOT')
```

### With Scheduler

```python
from lottery_scheduler import LotteryScheduler
from prediction_tracker import PredictionTracker

scheduler = LotteryScheduler()
tracker = PredictionTracker()

# Schedule to fetch and compare every 12 hours
scheduler.enable_all_lotteries(interval_hours=12)
```

### Complete Workflow

```python
# 1. Generate predictions (using existing system)
from ultra_lottery_helper import generate_candidates, GAMES

df, msg = _load_all_history('EUROJACKPOT')
candidates, msg = generate_candidates(df, 'EUROJACKPOT', config)

# 2. Save top prediction
tracker = PredictionTracker()
top_prediction = candidates[0][0]  # Get top candidate
tracker.save_prediction('EUROJACKPOT', '2026-01-25', top_prediction)

# 3. Wait for draw...

# 4. Fetch results
fetcher = LotteryDataFetcher()
fetcher.fetch_lottery_data('EUROJACKPOT')

# 5. Auto-compare
results = tracker.auto_compare_latest('EUROJACKPOT')
print(results)

# 6. View statistics
stats = tracker.get_accuracy_stats()
print(stats)
```

## Performance Considerations

- **Storage**: JSON files are lightweight (~1KB per 100 predictions)
- **Memory**: Minimal, loads only what's needed
- **Speed**: Fast lookups and comparisons (<100ms for most operations)

## Future Enhancements

Planned features:
- Ensemble prediction tracking (multiple predictions per draw)
- Model performance comparison (which ML model performs best)
- Confidence-weighted accuracy metrics
- Export to various formats (Excel, PDF reports)
- Web dashboard for visualization
- Email/SMS notifications for wins
- Historical backtesting on past data

## License

MIT License - Same as main project

## Support

For issues:
1. Check `data/history/predictions.json` exists
2. Verify prediction format matches lottery specs
3. Ensure draw dates are in correct format (YYYY-MM-DD)
4. Check logs for error messages
