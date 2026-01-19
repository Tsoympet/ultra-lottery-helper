#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lottery Prediction Tracker - Prediction Comparison and Validation System
Tracks predictions, compares with actual results, and provides accuracy metrics.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Add src to path if needed
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.ultra_lottery_helper import GAMES, LOTTERY_METADATA, _load_all_history
    from src.utils import get_logger, load_json, save_json
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    GAMES = {}
    LOTTERY_METADATA = {}
    print(f"Warning: Core modules not available: {e}")
    # Fallback logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    get_logger = lambda name: logging.getLogger(name)
    # Simple fallback implementations
    def load_json(path, default=None, logger=None):
        import json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default
    def save_json(path, data, atomic=True, logger=None):
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

logger = get_logger('prediction_tracker')


class PredictionTracker:
    """
    Tracks lottery predictions and compares them with actual draw results.
    Provides accuracy metrics and performance analysis.
    """
    
    def __init__(self, data_root: str = None):
        """
        Initialize the prediction tracker.
        
        Args:
            data_root: Root directory for data storage (default: data/history)
        """
        self.data_root = data_root or "data/history"
        self.predictions_file = os.path.join(self.data_root, "predictions.json")
        self.results_file = os.path.join(self.data_root, "prediction_results.json")
        self.predictions = self._load_predictions()
        self.results = self._load_results()
    
    def _load_predictions(self) -> Dict:
        """Load saved predictions from file."""
        return load_json(self.predictions_file, default={}, logger=logger)
    
    def _save_predictions(self):
        """Save predictions to file with atomic write."""
        save_json(self.predictions_file, self.predictions, atomic=True, logger=logger)
    
    def _load_results(self) -> Dict:
        """Load prediction results from file."""
        return load_json(self.results_file, default={}, logger=logger)
    
    def _save_results(self):
        """Save prediction results to file with atomic write."""
        save_json(self.results_file, self.results, atomic=True, logger=logger)
    
    def save_prediction(self, game: str, draw_date: str, 
                       predicted_numbers: List[int], 
                       metadata: Dict = None) -> str:
        """
        Save a prediction for a specific lottery and draw date.
        
        Args:
            game: Lottery name (e.g., "TZOKER", "EUROJACKPOT")
            draw_date: Expected draw date (ISO format: YYYY-MM-DD)
            predicted_numbers: List of predicted numbers
            metadata: Optional metadata (confidence scores, method used, etc.)
        
        Returns:
            Prediction ID
        """
        if game not in GAMES:
            raise ValueError(f"Unknown lottery: {game}")
        
        # Generate prediction ID
        pred_id = f"{game}_{draw_date}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Validate numbers
        spec = GAMES[game]
        if len(predicted_numbers) != (spec.main_pick + spec.sec_pick):
            raise ValueError(f"Expected {spec.main_pick + spec.sec_pick} numbers, got {len(predicted_numbers)}")
        
        # Create prediction entry
        prediction = {
            "id": pred_id,
            "game": game,
            "draw_date": draw_date,
            "predicted_numbers": predicted_numbers,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "status": "pending"  # pending, matched, unmatched
        }
        
        # Store prediction
        if game not in self.predictions:
            self.predictions[game] = []
        
        self.predictions[game].append(prediction)
        self._save_predictions()
        
        logger.info(f"Saved prediction {pred_id}: {predicted_numbers}")
        return pred_id
    
    def compare_with_draw(self, game: str, draw_date: str, 
                         actual_numbers: List[int]) -> Dict:
        """
        Compare predictions with actual draw results.
        
        Args:
            game: Lottery name
            draw_date: Draw date (ISO format)
            actual_numbers: Actual drawn numbers
        
        Returns:
            Comparison results dictionary
        """
        if game not in self.predictions:
            return {"message": f"No predictions found for {game}"}
        
        # Find predictions for this draw
        matching_preds = [
            p for p in self.predictions[game]
            if p["draw_date"] == draw_date and p["status"] == "pending"
        ]
        
        if not matching_preds:
            return {"message": f"No pending predictions for {game} on {draw_date}"}
        
        results = []
        spec = GAMES[game]
        
        for pred in matching_preds:
            predicted = set(pred["predicted_numbers"])
            actual = set(actual_numbers)
            
            # Calculate matches
            matches = predicted.intersection(actual)
            match_count = len(matches)
            
            # Separate main and secondary numbers if applicable
            main_matches = 0
            sec_matches = 0
            
            if spec.sec_pick > 0:
                # Split numbers
                pred_main = set(pred["predicted_numbers"][:spec.main_pick])
                pred_sec = set(pred["predicted_numbers"][spec.main_pick:])
                actual_main = set(actual_numbers[:spec.main_pick])
                actual_sec = set(actual_numbers[spec.main_pick:])
                
                main_matches = len(pred_main.intersection(actual_main))
                sec_matches = len(pred_sec.intersection(actual_sec))
            else:
                main_matches = match_count
            
            # Calculate accuracy
            accuracy = (match_count / len(actual)) * 100
            
            # Determine prize tier (simplified - would need lottery-specific rules)
            prize_tier = self._determine_prize_tier(game, main_matches, sec_matches)
            
            result = {
                "prediction_id": pred["id"],
                "draw_date": draw_date,
                "predicted_numbers": pred["predicted_numbers"],
                "actual_numbers": actual_numbers,
                "matched_numbers": list(matches),
                "match_count": match_count,
                "main_matches": main_matches,
                "sec_matches": sec_matches,
                "accuracy": round(accuracy, 2),
                "prize_tier": prize_tier,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            
            # Update prediction status
            pred["status"] = "matched"
            
            # Store result
            if game not in self.results:
                self.results[game] = []
            self.results[game].append(result)
        
        self._save_predictions()
        self._save_results()
        
        return {
            "game": game,
            "draw_date": draw_date,
            "predictions_evaluated": len(results),
            "results": results
        }
    
    def _determine_prize_tier(self, game: str, main_matches: int, 
                             sec_matches: int) -> str:
        """
        Determine prize tier based on matches (simplified).
        Real implementation would need lottery-specific rules.
        """
        spec = GAMES[game]
        
        # Jackpot
        if main_matches == spec.main_pick and sec_matches == spec.sec_pick:
            return "Jackpot"
        
        # High tier (5+ main matches)
        if main_matches >= 5:
            return f"Tier 2 ({main_matches} matches)"
        
        # Mid tier (4 matches)
        if main_matches == 4:
            return f"Tier 3 ({main_matches} matches)"
        
        # Low tier (3 matches)
        if main_matches == 3:
            return f"Tier 4 ({main_matches} matches)"
        
        # No prize
        return "No prize"
    
    def auto_compare_latest(self, game: str = None) -> Dict:
        """
        Automatically compare pending predictions with latest draw results.
        
        Args:
            game: Specific lottery to check, or None for all
        
        Returns:
            Dictionary with comparison results
        """
        games_to_check = [game] if game else list(self.predictions.keys())
        all_results = {}
        
        for g in games_to_check:
            if g not in GAMES:
                continue
            
            # Load latest draws
            try:
                df, msg = _load_all_history(g)
                if df.empty:
                    logger.warning(f"{g}: No historical data available")
                    continue
                
                # Get latest draw
                latest_draw = df.iloc[0] if len(df) > 0 else None
                if latest_draw is None:
                    continue
                
                # Extract date and numbers
                draw_date = latest_draw.get('date', '').strftime('%Y-%m-%d') if hasattr(latest_draw.get('date'), 'strftime') else str(latest_draw.get('date', ''))
                
                spec = GAMES[g]
                actual_numbers = []
                for col in spec.cols:
                    if col in latest_draw:
                        actual_numbers.append(int(latest_draw[col]))
                
                if actual_numbers:
                    result = self.compare_with_draw(g, draw_date, actual_numbers)
                    all_results[g] = result
                    
            except Exception as e:
                logger.error(f"Error processing {g}: {e}")
                all_results[g] = {"error": str(e)}
        
        return all_results
    
    def get_accuracy_stats(self, game: str = None, 
                          days_back: int = 30) -> pd.DataFrame:
        """
        Get accuracy statistics for predictions.
        
        Args:
            game: Specific lottery, or None for all
            days_back: Number of days to look back
        
        Returns:
            DataFrame with accuracy statistics
        """
        games_to_analyze = [game] if game else list(self.results.keys())
        stats_data = []
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for g in games_to_analyze:
            if g not in self.results:
                continue
            
            game_results = self.results[g]
            
            # Filter by date
            recent_results = [
                r for r in game_results
                if datetime.fromisoformat(r['timestamp']) >= cutoff_date
            ]
            
            if not recent_results:
                continue
            
            # Calculate statistics
            total_preds = len(recent_results)
            avg_matches = np.mean([r['match_count'] for r in recent_results])
            avg_accuracy = np.mean([r['accuracy'] for r in recent_results])
            max_matches = max([r['match_count'] for r in recent_results])
            
            # Count prize winners
            prize_wins = sum(1 for r in recent_results if r['prize_tier'] != "No prize")
            
            stats_data.append({
                'Lottery': LOTTERY_METADATA.get(g, {}).get('display_name', g),
                'Predictions': total_preds,
                'Avg Matches': round(avg_matches, 2),
                'Max Matches': max_matches,
                'Avg Accuracy %': round(avg_accuracy, 2),
                'Prize Wins': prize_wins,
                'Win Rate %': round((prize_wins / total_preds) * 100, 2) if total_preds > 0 else 0
            })
        
        return pd.DataFrame(stats_data)
    
    def get_pending_predictions(self, game: str = None) -> pd.DataFrame:
        """
        Get list of pending predictions.
        
        Args:
            game: Specific lottery, or None for all
        
        Returns:
            DataFrame with pending predictions
        """
        games_to_check = [game] if game else list(self.predictions.keys())
        pending_data = []
        
        for g in games_to_check:
            if g not in self.predictions:
                continue
            
            for pred in self.predictions[g]:
                if pred['status'] == 'pending':
                    pending_data.append({
                        'Lottery': LOTTERY_METADATA.get(g, {}).get('display_name', g),
                        'Draw Date': pred['draw_date'],
                        'Numbers': ', '.join(map(str, pred['predicted_numbers'])),
                        'Created': pred['created_at'],
                        'ID': pred['id']
                    })
        
        return pd.DataFrame(pending_data)
    
    def clear_old_predictions(self, days_old: int = 90):
        """
        Clear old matched predictions to save space.
        
        Args:
            days_old: Remove matched predictions older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        removed_count = 0
        
        for game in list(self.predictions.keys()):
            original_count = len(self.predictions[game])
            
            # Keep only recent or pending predictions
            self.predictions[game] = [
                p for p in self.predictions[game]
                if p['status'] == 'pending' or
                datetime.fromisoformat(p['created_at']) >= cutoff_date
            ]
            
            removed_count += original_count - len(self.predictions[game])
        
        if removed_count > 0:
            self._save_predictions()
            logger.info(f"Removed {removed_count} old predictions")
        
        return removed_count


def main():
    """Command-line interface for prediction tracker."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Lottery Prediction Tracker')
    parser.add_argument('--save', type=str, metavar='GAME',
                       help='Save a prediction for a lottery')
    parser.add_argument('--numbers', type=str,
                       help='Comma-separated predicted numbers')
    parser.add_argument('--draw-date', type=str,
                       help='Expected draw date (YYYY-MM-DD)')
    parser.add_argument('--compare', type=str, metavar='GAME',
                       help='Compare predictions with actual results')
    parser.add_argument('--actual', type=str,
                       help='Comma-separated actual numbers')
    parser.add_argument('--auto-compare', action='store_true',
                       help='Auto-compare all pending predictions with latest draws')
    parser.add_argument('--stats', action='store_true',
                       help='Show accuracy statistics')
    parser.add_argument('--pending', action='store_true',
                       help='Show pending predictions')
    parser.add_argument('--game', type=str,
                       help='Filter by specific lottery')
    parser.add_argument('--days', type=int, default=30,
                       help='Days to look back for stats (default: 30)')
    
    args = parser.parse_args()
    
    tracker = PredictionTracker()
    
    if args.save:
        if not args.numbers or not args.draw_date:
            print("Error: --numbers and --draw-date required for saving prediction")
            sys.exit(1)
        
        numbers = [int(x.strip()) for x in args.numbers.split(',')]
        pred_id = tracker.save_prediction(args.save, args.draw_date, numbers)
        print(f"✓ Prediction saved: {pred_id}")
        print(f"  Lottery: {args.save}")
        print(f"  Draw Date: {args.draw_date}")
        print(f"  Numbers: {numbers}")
    
    elif args.compare:
        if not args.actual or not args.draw_date:
            print("Error: --actual and --draw-date required for comparison")
            sys.exit(1)
        
        actual = [int(x.strip()) for x in args.actual.split(',')]
        result = tracker.compare_with_draw(args.compare, args.draw_date, actual)
        
        print(f"\n=== Prediction Comparison: {args.compare} ===")
        print(f"Draw Date: {args.draw_date}")
        print(f"Actual Numbers: {actual}")
        print(f"\nEvaluated {result.get('predictions_evaluated', 0)} prediction(s)")
        
        for r in result.get('results', []):
            print(f"\nPrediction: {r['predicted_numbers']}")
            print(f"Matches: {r['matched_numbers']} ({r['match_count']} total)")
            print(f"Accuracy: {r['accuracy']}%")
            print(f"Prize Tier: {r['prize_tier']}")
    
    elif args.auto_compare:
        print("Running auto-comparison...")
        results = tracker.auto_compare_latest(args.game)
        
        for game, result in results.items():
            if 'error' in result:
                print(f"\n{game}: Error - {result['error']}")
            elif 'message' in result:
                print(f"\n{game}: {result['message']}")
            else:
                print(f"\n{game}: Evaluated {result.get('predictions_evaluated', 0)} prediction(s)")
    
    elif args.stats:
        print("\n=== Prediction Accuracy Statistics ===")
        stats_df = tracker.get_accuracy_stats(args.game, args.days)
        
        if stats_df.empty:
            print("No statistics available")
        else:
            print(stats_df.to_string(index=False))
    
    elif args.pending:
        print("\n=== Pending Predictions ===")
        pending_df = tracker.get_pending_predictions(args.game)
        
        if pending_df.empty:
            print("No pending predictions")
        else:
            print(pending_df.to_string(index=False))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
