#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lottery Data Scheduler - Automated Scheduling System
Manages periodic fetching of lottery draw results with configurable schedules.
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path if needed
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.lottery_data_fetcher import LotteryDataFetcher
    from src.ultra_lottery_helper import GAMES, LOTTERY_METADATA
    from src.utils import get_logger, load_json, save_json
    from src.config import ValidationConfig
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    _import_error_msg = str(e)
    # Create dummy classes for when core is not available
    class LotteryDataFetcher:
        def __init__(self, data_root=None):
            self.data_root = data_root or "data/history"
            raise ImportError(f"LotteryDataFetcher not available: {_import_error_msg}")
    
    GAMES = {}
    LOTTERY_METADATA = {}
    # Fallback logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    get_logger = lambda name: logging.getLogger(name)
    print(f"Warning: Core modules not available: {e}")

logger = get_logger('lottery_scheduler')

# Try to import APScheduler for advanced scheduling
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    BackgroundScheduler = None
    IntervalTrigger = None
    CronTrigger = None


class LotteryScheduler:
    """
    Scheduler for automated lottery data fetching.
    Supports simple interval-based and cron-based scheduling.
    """
    
    def __init__(self, data_root: str = None, use_apscheduler: bool = None):
        """
        Initialize the scheduler.
        
        Args:
            data_root: Root directory for data storage
            use_apscheduler: Force use of APScheduler (default: auto-detect)
        """
        self.fetcher = LotteryDataFetcher(data_root=data_root)
        self.config_file = os.path.join(self.fetcher.data_root, "schedule_config.json")
        self.schedule_config = self._load_schedule_config()
        self.running = False
        
        # Determine scheduler backend
        if use_apscheduler is None:
            self.use_apscheduler = APSCHEDULER_AVAILABLE
        else:
            self.use_apscheduler = use_apscheduler and APSCHEDULER_AVAILABLE
        
        if self.use_apscheduler:
            self.scheduler = BackgroundScheduler()
            logger.info("Using APScheduler backend")
        else:
            self.scheduler = None
            logger.info("Using simple interval-based scheduling")
    
    def _load_schedule_config(self) -> Dict:
        """Load scheduling configuration from file."""
        return load_json(self.config_file, default=self._get_default_config(), logger=logger)
    
    def _save_schedule_config(self):
        """Save scheduling configuration to file with atomic write."""
        save_json(self.config_file, self.schedule_config, atomic=True, logger=logger)
    
    def _get_default_config(self) -> Dict:
        """Get default scheduling configuration."""
        return {
            "enabled": False,
            "default_interval_hours": 12,
            "schedules": {
                # Example: lottery_name: {"interval_hours": 12, "enabled": True}
            },
            "global_settings": {
                "respect_cache": True,  # Respect 6-hour minimum cache
                "batch_delay_seconds": 2,  # Delay between lottery fetches
                "max_retries": 3,
                "retry_delay_minutes": 5
            }
        }
    
    def add_schedule(self, game: str, interval_hours: int = 12, 
                     cron_expression: str = None, enabled: bool = True,
                     raise_on_unknown: bool = False) -> bool:
        """
        Add or update schedule for a lottery.
        
        Args:
            game: Lottery name
            interval_hours: Hours between fetches (used if no cron)
            cron_expression: Cron expression (e.g., "0 */12 * * *") - requires APScheduler
            enabled: Whether schedule is active
            raise_on_unknown: If True, raise ValueError on unknown lottery instead of returning False

        Returns:
            True if schedule was added, False otherwise.
        """
        # Validate game
        if game not in GAMES:
            msg = f"Unknown lottery: {game}"
            if raise_on_unknown:
                raise ValueError(msg)
            logger.error(msg)
            return False
        
        # Validate interval_hours using config
        if interval_hours is not None:
            if not ValidationConfig.MIN_INTERVAL_HOURS <= interval_hours <= ValidationConfig.MAX_INTERVAL_HOURS:
                raise ValueError(
                    f"interval_hours must be between {ValidationConfig.MIN_INTERVAL_HOURS} "
                    f"and {ValidationConfig.MAX_INTERVAL_HOURS}, got {interval_hours}"
                )
        
        schedule_entry = {
            "enabled": enabled,
            "interval_hours": interval_hours or self.schedule_config["default_interval_hours"]
        }
        
        if cron_expression and self.use_apscheduler:
            schedule_entry["cron"] = cron_expression
        
        self.schedule_config["schedules"][game] = schedule_entry
        self._save_schedule_config()
        logger.info(f"Schedule added for {game}: {schedule_entry}")
        return True
    
    def remove_schedule(self, game: str):
        """Remove schedule for a lottery."""
        if game in self.schedule_config["schedules"]:
            del self.schedule_config["schedules"][game]
            self._save_schedule_config()
            logger.info(f"Schedule removed for {game}")
    
    def enable_all_lotteries(self, interval_hours: int = 12):
        """Enable automatic fetching for all lotteries."""
        for game in GAMES.keys():
            self.add_schedule(game, interval_hours=interval_hours, enabled=True)
        self.schedule_config["enabled"] = True
        self._save_schedule_config()
        logger.info(f"Enabled all lotteries with {interval_hours}h interval")
    
    def _fetch_with_retry(self, game: str) -> bool:
        """Fetch lottery data with retry logic."""
        max_retries = self.schedule_config["global_settings"]["max_retries"]
        retry_delay = self.schedule_config["global_settings"]["retry_delay_minutes"]
        respect_cache = self.schedule_config["global_settings"]["respect_cache"]
        
        for attempt in range(max_retries):
            try:
                success, msg = self.fetcher.fetch_lottery_data(
                    game, 
                    force=not respect_cache
                )
                logger.info(f"{game}: {msg}")
                return success
            except Exception as e:
                logger.error(f"{game} fetch failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * 60)
        
        return False
    
    def _run_scheduled_fetch(self, game: str):
        """Run a scheduled fetch for a specific lottery."""
        logger.info(f"Running scheduled fetch for {game}")
        self._fetch_with_retry(game)
    
    def _run_all_scheduled(self):
        """Run fetch for all scheduled lotteries."""
        delay = self.schedule_config["global_settings"]["batch_delay_seconds"]
        
        for game, config in self.schedule_config["schedules"].items():
            if config.get("enabled", False):
                self._fetch_with_retry(game)
                time.sleep(delay)
    
    def start_simple_scheduler(self):
        """
        Start simple interval-based scheduler (no APScheduler required).
        Runs in foreground - use Ctrl+C to stop.
        """
        if not self.schedule_config.get("enabled", False):
            logger.warning("Scheduler is disabled in config. Enable with enable_all_lotteries()")
            return
        
        logger.info("Starting simple scheduler (Ctrl+C to stop)")
        self.running = True
        
        # Track next fetch time for each lottery
        next_fetch = {}
        for game, config in self.schedule_config["schedules"].items():
            if config.get("enabled", False):
                next_fetch[game] = datetime.now()
        
        try:
            while self.running:
                now = datetime.now()
                
                for game, next_time in next_fetch.items():
                    if now >= next_time:
                        config = self.schedule_config["schedules"][game]
                        if config.get("enabled", False):
                            self._run_scheduled_fetch(game)
                            
                            # Schedule next fetch
                            interval = timedelta(hours=config.get("interval_hours", 12))
                            next_fetch[game] = now + interval
                            logger.info(f"{game}: Next fetch at {next_fetch[game]}")
                
                # Sleep for 1 minute before checking again
                time.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            self.running = False
    
    def start_apscheduler(self):
        """
        Start APScheduler-based scheduler (requires apscheduler package).
        Runs in background - call stop() to stop.
        """
        if not self.use_apscheduler:
            logger.error("APScheduler not available. Install with: pip install apscheduler")
            return
        
        if not self.schedule_config.get("enabled", False):
            logger.warning("Scheduler is disabled in config. Enable with enable_all_lotteries()")
            return
        
        logger.info("Starting APScheduler")
        
        # Add jobs for each scheduled lottery
        for game, config in self.schedule_config["schedules"].items():
            if not config.get("enabled", False):
                continue
            
            if "cron" in config:
                # Use cron trigger
                trigger = CronTrigger.from_crontab(config["cron"])
                self.scheduler.add_job(
                    func=self._run_scheduled_fetch,
                    trigger=trigger,
                    args=[game],
                    id=f"fetch_{game}",
                    name=f"Fetch {game}",
                    replace_existing=True
                )
                logger.info(f"Added cron job for {game}: {config['cron']}")
            else:
                # Use interval trigger
                interval = config.get("interval_hours", 12)
                trigger = IntervalTrigger(hours=interval)
                self.scheduler.add_job(
                    func=self._run_scheduled_fetch,
                    trigger=trigger,
                    args=[game],
                    id=f"fetch_{game}",
                    name=f"Fetch {game}",
                    replace_existing=True
                )
                logger.info(f"Added interval job for {game}: every {interval} hours")
        
        self.scheduler.start()
        self.running = True
        logger.info("APScheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self.use_apscheduler and self.scheduler:
            self.scheduler.shutdown()
            logger.info("APScheduler stopped")
        else:
            logger.info("Simple scheduler stopped")
    
    def get_status(self) -> Dict:
        """Get scheduler status."""
        status = {
            "enabled": self.schedule_config.get("enabled", False),
            "backend": "APScheduler" if self.use_apscheduler else "Simple",
            "running": self.running,
            "scheduled_lotteries": []
        }
        
        for game, config in self.schedule_config["schedules"].items():
            if config.get("enabled", False):
                lottery_info = {
                    "game": game,
                    "display_name": LOTTERY_METADATA.get(game, {}).get("display_name", game),
                    "interval_hours": config.get("interval_hours", "N/A"),
                    "cron": config.get("cron", "N/A")
                }
                status["scheduled_lotteries"].append(lottery_info)
        
        return status


def main():
    """Command-line interface for scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Lottery Data Scheduler')
    parser.add_argument('--enable-all', action='store_true', 
                       help='Enable scheduling for all lotteries')
    parser.add_argument('--interval', type=int, default=12,
                       help='Default interval in hours (default: 12)')
    parser.add_argument('--add', type=str, metavar='LOTTERY',
                       help='Add specific lottery to schedule')
    parser.add_argument('--remove', type=str, metavar='LOTTERY',
                       help='Remove lottery from schedule')
    parser.add_argument('--start', action='store_true',
                       help='Start the scheduler')
    parser.add_argument('--status', action='store_true',
                       help='Show scheduler status')
    parser.add_argument('--use-apscheduler', action='store_true',
                       help='Use APScheduler (requires: pip install apscheduler)')
    
    args = parser.parse_args()
    
    scheduler = LotteryScheduler(use_apscheduler=args.use_apscheduler)
    
    if args.enable_all:
        scheduler.enable_all_lotteries(interval_hours=args.interval)
        print(f"✓ Enabled all lotteries with {args.interval}h interval")
        print(f"✓ Configuration saved to: {scheduler.config_file}")
        print("\nTo start the scheduler, run:")
        print(f"  python {__file__} --start")
    
    elif args.add:
        scheduler.add_schedule(args.add, interval_hours=args.interval)
        print(f"✓ Added {args.add} to schedule (every {args.interval} hours)")
    
    elif args.remove:
        scheduler.remove_schedule(args.remove)
        print(f"✓ Removed {args.remove} from schedule")
    
    elif args.status:
        status = scheduler.get_status()
        print("\n=== Lottery Scheduler Status ===")
        print(f"Enabled: {status['enabled']}")
        print(f"Backend: {status['backend']}")
        print(f"Running: {status['running']}")
        print(f"\nScheduled Lotteries ({len(status['scheduled_lotteries'])}):")
        for lottery in status['scheduled_lotteries']:
            interval = lottery['interval_hours']
            cron = lottery['cron']
            schedule_str = f"every {interval}h" if cron == "N/A" else f"cron: {cron}"
            print(f"  - {lottery['display_name']:25} ({schedule_str})")
        print("=" * 40)
    
    elif args.start:
        status = scheduler.get_status()
        if not status['enabled']:
            print("✗ Scheduler is not enabled!")
            print("Enable with: python {} --enable-all".format(__file__))
            sys.exit(1)
        
        print(f"Starting scheduler with {len(status['scheduled_lotteries'])} lotteries...")
        print("Press Ctrl+C to stop\n")
        
        if scheduler.use_apscheduler:
            scheduler.start_apscheduler()
            try:
                # Keep main thread alive
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                scheduler.stop()
        else:
            scheduler.start_simple_scheduler()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
