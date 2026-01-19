import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QGridLayout, QLineEdit
)
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtCore import Qt, QSize, Slot, QTimer

# ----- Version detection helper -----
def get_app_version() -> str:
    try:
        try:
            from ultra_lottery_helper import __version__ as v
            if isinstance(v, str) and v.strip():
                return v.strip()
        except Exception:
            pass
        import re, os
        from pathlib import Path as _P
        HERE = _P(__file__).resolve().parent
        root = HERE.parent
        pyproj = root / "pyproject.toml"
        if pyproj.exists():
            t = pyproj.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r'(?m)^\s*version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', t)
            if m:
                return m.group(1)
        vd = root / "version_detected.txt"
        if vd.exists():
            v = vd.read_text(encoding="utf-8", errors="ignore").strip()
            if v:
                return v
    except Exception:
        pass
    return "6.3.0"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Oracle Lottery Predictor — Desktop v{get_app_version()}")
        self.setMinimumSize(800, 600)

        central = QWidget()
        root = QVBoxLayout()
        central.setLayout(root)
        self.setCentralWidget(central)

        # UI groups
        root.addWidget(self._build_predict_group())
        root.addWidget(self._build_learn_group())
        root.addWidget(self._build_settings_group())

        # Status label
        self.status_label = QLabel("Status: ready")
        root.addWidget(self.status_label)

        # Do initial status update
        try:
            self._update_status()
        except Exception:
            pass

    def _build_ui(self):
        # Legacy compatibility; integrated into __init__
        root = QVBoxLayout()
        row1 = QHBoxLayout()
        btn_predict = QPushButton("Predict")
        btn_learn = QPushButton("Learn Now")
        row1.addWidget(btn_predict)
        row1.addWidget(btn_learn)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        self.last_result_label = QLabel("Last prediction: -")
        row2.addWidget(self.last_result_label)
        root.addLayout(row2)

        try:
            self._update_status()
        except Exception:
            pass

        return root

    def _build_predict_group(self):
        g = QGroupBox("Prediction")
        lay = QHBoxLayout()
        g.setLayout(lay)
        self.btn_predict = QPushButton("Predict")
        lay.addWidget(self.btn_predict)
        self.lbl_predict = QLabel("Last: -")
        lay.addWidget(self.lbl_predict)
        return g

    def _build_learn_group(self):
        g = QGroupBox("Learning")
        lay = QHBoxLayout()
        g.setLayout(lay)
        self.btn_learn = QPushButton("Learn Now")
        lay.addWidget(self.btn_learn)
        self.lbl_learn = QLabel("Learning status: -")
        lay.addWidget(self.lbl_learn)
        return g

    def _build_settings_group(self):
        g = QGroupBox("Settings")
        lay = QGridLayout()
        g.setLayout(lay)
        lay.addWidget(QLabel("Parameter A:"), 0, 0)
        self.txt_param_a = QLineEdit()
        lay.addWidget(self.txt_param_a, 0, 1)
        return g

    def _update_status(self):
        # Example status updater
        self.status_label.setText("Status: updated")


def main():
    app = QApplication(sys.argv)
    from PySide6.QtWidgets import QSplashScreen
    from pathlib import Path as _P
    HERE = _P(__file__).resolve().parent
    pix = QPixmap(str((HERE.parent / "assets" / "splash.png")))
    splash = QSplashScreen(pix)
    splash.show()
    app.processEvents()
    win = MainWindow()
    win.show()
    QTimer.singleShot(1200, splash.close)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
