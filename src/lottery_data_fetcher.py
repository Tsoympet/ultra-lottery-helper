#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Lottery Data Fetcher - Live Feed Module
Fetches latest draw results from official lottery sources and stores them locally.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add src to path if needed
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.ultra_lottery_helper import (
        DATA_ROOT,
        GAMES,
        LOTTERY_METADATA,
        _game_path,
        fetch_online_history,
    )
    from src.utils import get_logger, load_json, save_json
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("Warning: Core lottery helper not available. Some features may not work.")
    # Fallback
    import logging
    get_logger = lambda name: logging.getLogger(name)

logger = get_logger('lottery_data_fetcher')


class LotteryDataFetcher:
    """
    Automated fetcher for lottery draw results.
    Supports live feed updates and local storage.
    """
    
    def __init__(self, data_root: str = None):
        """Initialize the data fetcher."""
        self.data_root = data_root or DATA_ROOT
        self.fetch_log_file = os.path.join(self.data_root, "fetch_log.json")
        self.fetch_log = self._load_fetch_log()
    
    def _load_fetch_log(self) -> Dict:
        """Load the fetch log tracking when each lottery was last updated."""
        return load_json(self.fetch_log_file, default={}, logger=logger)
    
    def _save_fetch_log(self):
        """Save the fetch log with atomic write."""
        save_json(self.fetch_log_file, self.fetch_log, atomic=True, logger=logger)
    
    def fetch_lottery_data(self, game: str, force: bool = False) -> Tuple[bool, str]:
        """
        Fetch latest data for a specific lottery.
        
        Args:
            game: Lottery name (e.g., "TZOKER", "UK_NATIONAL_LOTTERY")
            force: Force fetch even if recently updated
        
        Returns:
            (success, message) tuple
            
        Raises:
            ValueError: If game is unknown
        """
        if game not in GAMES:
            raise ValueError(f"Unknown lottery: {game}")
        
        # Check if we need to fetch (avoid too frequent requests)
        last_fetch = self.fetch_log.get(game, {}).get('last_fetch')
        if last_fetch and not force:
            last_fetch_time = datetime.fromisoformat(last_fetch)
            hours_since = (datetime.now() - last_fetch_time).total_seconds() / 3600
            
            # Don't fetch if updated in last 6 hours
            if hours_since < 6:
                return False, f"{game}: Last fetched {hours_since:.1f} hours ago. Use force=True to override."
        
        # Fetch online data
        print(f"Fetching latest data for {game}...")
        df, msg = fetch_online_history(game)
        
        if df.empty:
            return False, f"{game}: {msg}"
        
        # Save to CSV with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_path = _game_path(game)
        os.makedirs(game_path, exist_ok=True)
        
        output_file = os.path.join(game_path, f"fetched_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        
        # Update fetch log
        self.fetch_log[game] = {
            'last_fetch': datetime.now().isoformat(),
            'rows_fetched': len(df),
            'file': output_file,
            'status': 'success'
        }
        self._save_fetch_log()
        
        return True, f"{game}: Fetched {len(df)} draws and saved to {output_file}"
    
    def fetch_all_lotteries(self, force: bool = False) -> Dict[str, Tuple[bool, str]]:
        """
        Fetch data for all supported lotteries.
        
        Args:
            force: Force fetch even if recently updated
        
        Returns:
            Dictionary mapping lottery name to (success, message) tuples
        """
        results = {}
        
        for game in GAMES.keys():
            success, msg = self.fetch_lottery_data(game, force=force)
            results[game] = (success, msg)
            print(msg)
            
            # Be respectful - add delay between requests
            if success:
                time.sleep(2)
        
        return results
    
    def get_fetch_status(self) -> pd.DataFrame:
        """
        Get status of all lottery data fetches.
        
        Returns:
            DataFrame with fetch status for each lottery
        """
        status_data = []
        
        for game in GAMES.keys():
            metadata = LOTTERY_METADATA.get(game, {})
            fetch_info = self.fetch_log.get(game, {})
            
            last_fetch = fetch_info.get('last_fetch', 'Never')
            if last_fetch != 'Never':
                try:
                    last_fetch_dt = datetime.fromisoformat(last_fetch)
                    hours_ago = (datetime.now() - last_fetch_dt).total_seconds() / 3600
                    last_fetch = f"{hours_ago:.1f}h ago"
                except (ValueError, TypeError):
                    # Keep original value if parsing fails
                    pass
            
            status_data.append({
                'Lottery': metadata.get('display_name', game),
                'Country': metadata.get('country', 'Unknown'),
                'Last Fetch': last_fetch,
                'Rows': fetch_info.get('rows_fetched', 0),
                'Status': fetch_info.get('status', 'Not fetched'),
                'Has Jackpot': metadata.get('has_jackpot', False)
            })
        
        return pd.DataFrame(status_data)
    
    def schedule_update(self, game: str, interval_hours: int = 24):
        """
        Schedule periodic updates for a lottery.
        Note: This is a simple implementation. For production, use a proper scheduler.
        
        Args:
            game: Lottery name
            interval_hours: Hours between updates
        """
        # This is a basic implementation
        # In production, you'd use APScheduler, Celery, or similar
        print(f"Scheduling {game} updates every {interval_hours} hours")
        print("Note: For production use, implement proper scheduling with APScheduler or cron")


def fetch_latest_draws(game: str = None, force: bool = False) -> Dict:
    """
    Convenience function to fetch latest draws.
    
    Args:
        game: Specific lottery to fetch, or None for all
        force: Force fetch even if recently updated
    
    Returns:
        Dictionary with fetch results
    """
    fetcher = LotteryDataFetcher()
    
    if game:
        success, msg = fetcher.fetch_lottery_data(game, force=force)
        return {game: {'success': success, 'message': msg}}
    else:
        results = fetcher.fetch_all_lotteries(force=force)
        return {g: {'success': s, 'message': m} for g, (s, m) in results.items()}


def main():
    """Command-line interface for data fetcher."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Lottery Data Fetcher - Live Feed')
    parser.add_argument('--game', type=str, help='Specific lottery to fetch (e.g., TZOKER)')
    parser.add_argument('--all', action='store_true', help='Fetch all lotteries')
    parser.add_argument('--force', action='store_true', help='Force fetch even if recently updated')
    parser.add_argument('--status', action='store_true', help='Show fetch status')
    
    args = parser.parse_args()
    
    fetcher = LotteryDataFetcher()
    
    if args.status:
        print("\nLottery Data Fetch Status:")
        print("=" * 80)
        status_df = fetcher.get_fetch_status()
        print(status_df.to_string(index=False))
        print("=" * 80)
    elif args.all:
        print("\nFetching data for all lotteries...")
        print("=" * 80)
        results = fetcher.fetch_all_lotteries(force=args.force)
        print("=" * 80)
        print(f"\nCompleted: {sum(1 for s, _ in results.values() if s)}/{len(results)} successful")
    elif args.game:
        success, msg = fetcher.fetch_lottery_data(args.game, force=args.force)
        print(msg)
        if not success:
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
