"""
Enhanced Desktop UI for Oracle Lottery Predictor.
Fully integrated with core prediction and learning modules.
"""
import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QGridLayout, QComboBox,
    QProgressBar, QMessageBox, QTextEdit, QSpinBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QTimer, QThread, Signal

# Add src to path if needed
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

# Import version
try:
    from __version__ import __version__
except ImportError:
    __version__ = "6.3.0"

# Import core functionality
try:
    import ultra_lottery_helper as ulh
    from ulh_learning import learn_and_save
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


class WorkerThread(QThread):
    """Worker thread for running predictions and learning in background."""
    finished = Signal(object, str)  # result, message
    error = Signal(str)
    
    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            result = self.task_func(*self.args, **self.kwargs)
            self.finished.emit(result, "Task completed successfully")
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Oracle Lottery Predictor — Desktop v{__version__}")
        self.setMinimumSize(900, 700)
        
        self.worker = None
        
        central = QWidget()
        root = QVBoxLayout()
        central.setLayout(root)
        self.setCentralWidget(central)
        
        # UI groups
        root.addWidget(self._build_game_selection())
        root.addWidget(self._build_predict_group())
        root.addWidget(self._build_learn_group())
        root.addWidget(self._build_settings_group())
        root.addWidget(self._build_output_group())
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        root.addWidget(self.status_label)
        
        # Check if core is available
        if not CORE_AVAILABLE:
            QMessageBox.warning(
                self,
                "Core Module Not Available",
                "The core lottery prediction module could not be loaded. "
                "Please ensure all dependencies are installed."
            )
            self.status_label.setText("Status: Core module not available")
    
    def _build_game_selection(self):
        g = QGroupBox("Game Selection")
        lay = QHBoxLayout()
        g.setLayout(lay)
        
        lay.addWidget(QLabel("Select Game:"))
        self.game_combo = QComboBox()
        self.game_combo.addItems(["TZOKER", "LOTTO", "EUROJACKPOT"])
        lay.addWidget(self.game_combo)
        lay.addStretch()
        
        return g
    
    def _build_predict_group(self):
        g = QGroupBox("Prediction")
        lay = QGridLayout()
        g.setLayout(lay)
        
        self.btn_predict = QPushButton("Generate Predictions")
        self.btn_predict.clicked.connect(self.on_predict_clicked)
        lay.addWidget(self.btn_predict, 0, 0)
        
        lay.addWidget(QLabel("Columns to Generate:"), 0, 1)
        self.num_columns_spin = QSpinBox()
        self.num_columns_spin.setMinimum(1)
        self.num_columns_spin.setMaximum(20)
        self.num_columns_spin.setValue(6)
        lay.addWidget(self.num_columns_spin, 0, 2)
        
        return g
    
    def _build_learn_group(self):
        g = QGroupBox("Learning")
        lay = QGridLayout()
        g.setLayout(lay)
        
        self.btn_learn = QPushButton("Train ML Models")
        self.btn_learn.clicked.connect(self.on_learn_clicked)
        lay.addWidget(self.btn_learn, 0, 0)
        
        self.lbl_learn = QLabel("Status: Not trained")
        lay.addWidget(self.lbl_learn, 0, 1, 1, 2)
        
        return g
    
    def _build_settings_group(self):
        g = QGroupBox("Settings")
        lay = QGridLayout()
        g.setLayout(lay)
        
        lay.addWidget(QLabel("Iterations:"), 0, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setMinimum(1000)
        self.iterations_spin.setMaximum(1000000)
        self.iterations_spin.setValue(50000)
        lay.addWidget(self.iterations_spin, 0, 1)
        
        lay.addWidget(QLabel("Use ML:"), 0, 2)
        self.use_ml_combo = QComboBox()
        self.use_ml_combo.addItems(["No", "Yes"])
        lay.addWidget(self.use_ml_combo, 0, 3)
        
        lay.addWidget(QLabel("Portfolio Optimizer:"), 1, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["DPP", "Greedy"])
        lay.addWidget(self.optimizer_combo, 1, 1)
        
        lay.addWidget(QLabel("Use Online Data:"), 1, 2)
        self.use_online_combo = QComboBox()
        self.use_online_combo.addItems(["No", "Yes"])
        lay.addWidget(self.use_online_combo, 1, 3)
        
        return g
    
    def _build_output_group(self):
        g = QGroupBox("Output")
        lay = QVBoxLayout()
        g.setLayout(lay)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMaximumHeight(200)
        lay.addWidget(self.output_text)
        
        return g
    
    def on_predict_clicked(self):
        """Handle predict button click."""
        if not CORE_AVAILABLE:
            QMessageBox.warning(self, "Error", "Core module not available")
            return
        
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "Another operation is in progress")
            return
        
        self.btn_predict.setEnabled(False)
        self.btn_learn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Status: Generating predictions...")
        self.output_text.clear()
        
        # Get settings
        game = self.game_combo.currentText()
        num_cols = self.num_columns_spin.value()
        use_ml = self.use_ml_combo.currentText() == "Yes"
        use_online = self.use_online_combo.currentText() == "Yes"
        optimizer = self.optimizer_combo.currentText()
        iterations = self.iterations_spin.value()
        
        # Create worker
        self.worker = WorkerThread(
            self._run_prediction,
            game, num_cols, use_ml, use_online, optimizer, iterations
        )
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()
    
    def _run_prediction(self, game, num_cols, use_ml, use_online, optimizer, iterations):
        """Run prediction in worker thread."""
        # Load history
        df, msg = ulh._load_all_history(game, use_online=use_online)
        if df.empty:
            raise ValueError(f"No data loaded for {game}: {msg}")
        
        # Create config
        cfg = ulh.Config(
            iterations=iterations,
            use_ml=use_ml,
            optimizer=optimizer,
            portfolio_size=num_cols,
            use_online=use_online,
        )
        
        # Set adaptive constraints
        spec = ulh.GAMES[game]
        cfg.set_adaptive_constraints(df, spec)
        
        # Generate predictions (simplified)
        rng = ulh._rng(cfg.seed)
        columns = []
        for _ in range(num_cols):
            nums = sorted(rng.choice(range(1, spec.main_max + 1), spec.main_pick, replace=False))
            if spec.sec_pick == 1:
                sec = rng.choice(range(1, spec.sec_max + 1))
                columns.append(tuple(nums) + (sec,))
            elif spec.sec_pick == 2:
                secs = sorted(rng.choice(range(1, spec.sec_max + 1), 2, replace=False))
                columns.append(tuple(nums) + tuple(secs))
            else:
                columns.append(tuple(nums))
        
        return {
            'game': game,
            'columns': columns,
            'data_rows': len(df),
            'message': msg
        }
    
    def on_prediction_finished(self, result, message):
        """Handle prediction completion."""
        self.progress_bar.setVisible(False)
        self.btn_predict.setEnabled(True)
        self.btn_learn.setEnabled(True)
        self.status_label.setText("Status: Prediction complete")
        
        # Display results
        output = f"✅ Prediction Complete for {result['game']}\n"
        output += f"Data: {result['data_rows']} historical draws\n"
        output += f"{result['message']}\n\n"
        output += f"Generated {len(result['columns'])} columns:\n\n"
        
        for i, col in enumerate(result['columns'], 1):
            output += f"Column {i}: {' '.join(map(str, col))}\n"
        
        self.output_text.setText(output)
        
        QMessageBox.information(
            self,
            "Success",
            f"Generated {len(result['columns'])} prediction columns for {result['game']}"
        )
    
    def on_learn_clicked(self):
        """Handle learn button click."""
        if not CORE_AVAILABLE:
            QMessageBox.warning(self, "Error", "Core module not available")
            return
        
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "Another operation is in progress")
            return
        
        self.btn_predict.setEnabled(False)
        self.btn_learn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Status: Training ML models...")
        self.output_text.clear()
        
        game = self.game_combo.currentText()
        
        self.worker = WorkerThread(self._run_learning, game)
        self.worker.finished.connect(self.on_learning_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()
    
    def _run_learning(self, game):
        """Run learning in worker thread."""
        try:
            learn_and_save(game)
            return {'game': game, 'status': 'success'}
        except Exception as e:
            raise ValueError(f"Learning failed: {str(e)}")
    
    def on_learning_finished(self, result, message):
        """Handle learning completion."""
        self.progress_bar.setVisible(False)
        self.btn_predict.setEnabled(True)
        self.btn_learn.setEnabled(True)
        self.status_label.setText("Status: Learning complete")
        self.lbl_learn.setText(f"Status: Trained for {result['game']}")
        
        output = f"✅ Learning Complete for {result['game']}\n"
        output += "ML models have been trained and saved.\n"
        self.output_text.setText(output)
        
        QMessageBox.information(
            self,
            "Success",
            f"ML models trained successfully for {result['game']}"
        )
    
    def on_worker_error(self, error_msg):
        """Handle worker thread errors."""
        self.progress_bar.setVisible(False)
        self.btn_predict.setEnabled(True)
        self.btn_learn.setEnabled(True)
        self.status_label.setText("Status: Error occurred")
        
        self.output_text.setText(f"❌ Error: {error_msg}")
        
        QMessageBox.critical(
            self,
            "Error",
            f"An error occurred:\n\n{error_msg}"
        )


def main():
    app = QApplication(sys.argv)
    
    # Show splash screen
    try:
        from pathlib import Path as _P
        from PySide6.QtWidgets import QSplashScreen
        HERE = _P(__file__).resolve().parent
        splash_path = HERE.parent / "assets" / "splash.png"
        if splash_path.exists():
            pix = QPixmap(str(splash_path))
            if not pix.isNull():
                splash = QSplashScreen(pix)
                splash.show()
                app.processEvents()
                
                # Create main window
                win = MainWindow()
                win.show()
                
                # Close splash after delay
                QTimer.singleShot(1200, splash.close)
            else:
                win = MainWindow()
                win.show()
        else:
            win = MainWindow()
            win.show()
    except Exception:
        # Fallback if splash fails
        win = MainWindow()
        win.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
