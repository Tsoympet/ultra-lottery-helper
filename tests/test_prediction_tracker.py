"""
Unit tests for prediction tracker module.
"""
import pytest
import os
import json
import sys
import tempfile
import shutil
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prediction_tracker import PredictionTracker


class TestPredictionTracker:
    """Test prediction tracker functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary data directory
        self.test_data_root = tempfile.mkdtemp()
        self.tracker = PredictionTracker(data_root=self.test_data_root)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_data_root):
            shutil.rmtree(self.test_data_root)
    
    def test_tracker_initialization(self):
        """Test that tracker initializes correctly."""
        assert self.tracker.data_root == self.test_data_root
        assert isinstance(self.tracker.predictions, dict)
        assert isinstance(self.tracker.results, dict)
    
    def test_save_prediction(self):
        """Test saving a prediction."""
        pred_id = self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[3, 12, 18, 24, 30, 15]
        )
        
        assert pred_id.startswith('TZOKER_2026-01-25_')
        assert 'TZOKER' in self.tracker.predictions
        assert len(self.tracker.predictions['TZOKER']) == 1
        
        pred = self.tracker.predictions['TZOKER'][0]
        assert pred['game'] == 'TZOKER'
        assert pred['draw_date'] == '2026-01-25'
        assert pred['predicted_numbers'] == [3, 12, 18, 24, 30, 15]
        assert pred['status'] == 'pending'
    
    def test_save_prediction_with_metadata(self):
        """Test saving prediction with metadata."""
        metadata = {'confidence': 0.85, 'method': 'ML'}
        
        pred_id = self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[3, 12, 18, 24, 30, 15],
            metadata=metadata
        )
        
        pred = self.tracker.predictions['TZOKER'][0]
        assert pred['metadata'] == metadata
    
    def test_save_prediction_invalid_game(self):
        """Test that invalid game raises error."""
        with pytest.raises(ValueError, match="Unknown lottery"):
            self.tracker.save_prediction(
                game='INVALID_GAME',
                draw_date='2026-01-25',
                predicted_numbers=[1, 2, 3]
            )
    
    def test_save_prediction_wrong_number_count(self):
        """Test that wrong number count raises error."""
        with pytest.raises(ValueError, match="Expected .* numbers"):
            # TZOKER needs 6 numbers, providing only 3
            self.tracker.save_prediction(
                game='TZOKER',
                draw_date='2026-01-25',
                predicted_numbers=[1, 2, 3]
            )
    
    def test_compare_with_draw(self):
        """Test comparing prediction with actual draw."""
        # Save prediction
        self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[3, 12, 18, 24, 30, 15]
        )
        
        # Compare with actual
        result = self.tracker.compare_with_draw(
            game='TZOKER',
            draw_date='2026-01-25',
            actual_numbers=[3, 12, 19, 24, 30, 16]
        )
        
        assert result['game'] == 'TZOKER'
        assert result['predictions_evaluated'] == 1
        assert len(result['results']) == 1
        
        comp = result['results'][0]
        assert comp['match_count'] == 4  # 3, 12, 24, 30 match
        assert set(comp['matched_numbers']) == {3, 12, 24, 30}
        assert comp['accuracy'] > 0
    
    def test_compare_no_pending_predictions(self):
        """Test comparison when no pending predictions exist."""
        result = self.tracker.compare_with_draw(
            game='TZOKER',
            draw_date='2026-01-25',
            actual_numbers=[1, 2, 3, 4, 5, 6]
        )
        
        assert 'message' in result
        assert 'No predictions found' in result['message'] or 'No pending predictions' in result['message']
    
    def test_prediction_persistence(self):
        """Test that predictions persist across instances."""
        # Save prediction
        self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[3, 12, 18, 24, 30, 15]
        )
        
        # Create new tracker instance
        tracker2 = PredictionTracker(data_root=self.test_data_root)
        
        # Should load the same predictions
        assert 'TZOKER' in tracker2.predictions
        assert len(tracker2.predictions['TZOKER']) == 1
    
    def test_results_persistence(self):
        """Test that results persist across instances."""
        # Save and compare
        self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[3, 12, 18, 24, 30, 15]
        )
        
        self.tracker.compare_with_draw(
            game='TZOKER',
            draw_date='2026-01-25',
            actual_numbers=[3, 12, 19, 24, 30, 16]
        )
        
        # Create new tracker instance
        tracker2 = PredictionTracker(data_root=self.test_data_root)
        
        # Should load the same results
        assert 'TZOKER' in tracker2.results
        assert len(tracker2.results['TZOKER']) == 1
    
    def test_get_pending_predictions(self):
        """Test getting pending predictions."""
        # Save two predictions
        self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[3, 12, 18, 24, 30, 15]
        )
        
        self.tracker.save_prediction(
            game='EUROJACKPOT',
            draw_date='2026-01-26',
            predicted_numbers=[5, 12, 18, 27, 33, 2, 8]
        )
        
        pending_df = self.tracker.get_pending_predictions()
        
        assert len(pending_df) == 2
        assert 'Lottery' in pending_df.columns
        assert 'Draw Date' in pending_df.columns
    
    def test_get_accuracy_stats_empty(self):
        """Test getting stats when no results exist."""
        stats_df = self.tracker.get_accuracy_stats()
        
        assert stats_df.empty
    
    def test_clear_old_predictions(self):
        """Test clearing old predictions."""
        # Create old prediction (manually set old date)
        old_pred = {
            "id": "TZOKER_2025-01-01_20250101000000",
            "game": "TZOKER",
            "draw_date": "2025-01-01",
            "predicted_numbers": [1, 2, 3, 4, 5, 6],
            "created_at": (datetime.now() - timedelta(days=100)).isoformat(),
            "metadata": {},
            "status": "matched"
        }
        
        self.tracker.predictions['TZOKER'] = [old_pred]
        self.tracker._save_predictions()
        
        # Clear old predictions
        removed = self.tracker.clear_old_predictions(days_old=90)
        
        assert removed == 1
        assert len(self.tracker.predictions.get('TZOKER', [])) == 0
    
    def test_multiple_predictions_same_draw(self):
        """Test multiple predictions for same draw."""
        # Save two predictions for same draw
        self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[3, 12, 18, 24, 30, 15]
        )
        
        self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[5, 10, 20, 25, 35, 18]
        )
        
        # Compare should evaluate both
        result = self.tracker.compare_with_draw(
            game='TZOKER',
            draw_date='2026-01-25',
            actual_numbers=[3, 12, 19, 24, 30, 16]
        )
        
        assert result['predictions_evaluated'] == 2
        assert len(result['results']) == 2


class TestPredictionComparison:
    """Test prediction comparison logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data_root = tempfile.mkdtemp()
        self.tracker = PredictionTracker(data_root=self.test_data_root)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_data_root):
            shutil.rmtree(self.test_data_root)
    
    def test_perfect_match(self):
        """Test perfect match (jackpot)."""
        self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[3, 12, 18, 24, 30, 15]
        )
        
        result = self.tracker.compare_with_draw(
            game='TZOKER',
            draw_date='2026-01-25',
            actual_numbers=[3, 12, 18, 24, 30, 15]
        )
        
        comp = result['results'][0]
        assert comp['match_count'] == 6
        assert comp['accuracy'] == 100.0
        assert 'Jackpot' in comp['prize_tier']
    
    def test_no_match(self):
        """Test no matches."""
        self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[1, 2, 3, 4, 5, 6]
        )
        
        result = self.tracker.compare_with_draw(
            game='TZOKER',
            draw_date='2026-01-25',
            actual_numbers=[7, 8, 9, 10, 11, 12]
        )
        
        comp = result['results'][0]
        assert comp['match_count'] == 0
        assert comp['accuracy'] == 0.0
        assert 'No prize' in comp['prize_tier']
    
    def test_partial_match(self):
        """Test partial matches."""
        self.tracker.save_prediction(
            game='TZOKER',
            draw_date='2026-01-25',
            predicted_numbers=[3, 12, 18, 24, 30, 15]
        )
        
        result = self.tracker.compare_with_draw(
            game='TZOKER',
            draw_date='2026-01-25',
            actual_numbers=[3, 12, 19, 25, 31, 16]
        )
        
        comp = result['results'][0]
        assert comp['match_count'] == 2
        assert 0 < comp['accuracy'] < 100


class TestIntegration:
    """Integration tests for prediction tracker."""
    
    def test_save_compare_workflow(self):
        """Test complete save-compare workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PredictionTracker(data_root=tmpdir)
            
            # Save prediction
            pred_id = tracker.save_prediction(
                game='TZOKER',
                draw_date='2026-01-25',
                predicted_numbers=[3, 12, 18, 24, 30, 15],
                metadata={'method': 'test'}
            )
            
            assert pred_id is not None
            
            # Compare
            result = tracker.compare_with_draw(
                game='TZOKER',
                draw_date='2026-01-25',
                actual_numbers=[3, 12, 19, 24, 30, 16]
            )
            
            assert result['predictions_evaluated'] == 1
            
            # Check results stored
            assert 'TZOKER' in tracker.results
            
            # Get stats
            stats_df = tracker.get_accuracy_stats()
            assert not stats_df.empty
    
    def test_json_file_creation(self):
        """Test that JSON files are created correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PredictionTracker(data_root=tmpdir)
            
            # Save prediction
            tracker.save_prediction(
                game='TZOKER',
                draw_date='2026-01-25',
                predicted_numbers=[3, 12, 18, 24, 30, 15]
            )
            
            # Check predictions file exists and is valid JSON
            pred_file = os.path.join(tmpdir, "predictions.json")
            assert os.path.exists(pred_file)
            
            with open(pred_file, 'r') as f:
                data = json.load(f)
            
            assert 'TZOKER' in data
            assert len(data['TZOKER']) == 1
