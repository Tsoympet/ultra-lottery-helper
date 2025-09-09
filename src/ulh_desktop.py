#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra Lottery Helper — Native Desktop UI (PySide6)
- Local offline app
- Uses ultra_lottery_helper.py (core) from the same folder (src)
- Tabs per game: TZOKER, LOTTO, EUROJACKPOT
- Portfolio prediction, plots, exports
"""

import os
import sys
import traceback
import runpy
import pathlib

# ---------------- Robust core import ----------------
HERE = pathlib.Path(__file__).resolve().parent  # .../src

# Ensure src is on sys.path (NOT the project root)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import ultra_lottery_helper as ulh
except ModuleNotFoundError:
    # Fallback: load module by path (dev + some PyInstaller layouts)
    core_path = HERE / "ultra_lottery_helper.py"
    if core_path.exists():
        mod_dict = runpy.run_path(str(core_path))
        class _NS: pass
        ulh = _NS()
        for k, v in mod_dict.items():
            setattr(ulh, k, v)
    else:
        raise

# ---------------- Qt / Matplotlib imports ----------------
try:
    from PySide6.QtCore import Qt, QSize, Slot
    from PySide6.QtGui import QAction, QIcon
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QCheckBox, QSpinBox, QGroupBox, QGridLayout,
        QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
    )
except Exception as e:
    print("PySide6 is required to run the desktop UI. Please install with:")
    print("  pip install PySide6 matplotlib")
    raise

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

# Bring names from core (ulh namespace)
GAMES = ulh.GAMES
Config = ulh.Config
_load_all_history = ulh._load_all_history
build_probs = ulh.build_probs
generate_candidates = ulh.generate_candidates
dpp_select = ulh.dpp_select
monte_carlo_risk = ulh.monte_carlo_risk
apply_ev_rerank = ulh.apply_ev_rerank
export_six_to_csv = ulh.export_six_to_csv
export_six_to_png = ulh.export_six_to_png
plot_frequency = ulh.plot_frequency
plot_recency = ulh.plot_recency
plot_last_digit = ulh.plot_last_digit
plot_pairs_heatmap = ulh.plot_pairs_heatmap
plot_odd_even = ulh.plot_odd_even
_quick_df_sig = ulh._quick_df_sig
OPAP_TICKET_PRICE_DEFAULTS = ulh.OPAP_TICKET_PRICE_DEFAULTS

# ---------------- Helpers ----------------

def error_box(msg: str, parent=None):
    QMessageBox.critical(parent, "Ultra Lottery Helper — Error", msg)

def info_box(msg: str, parent=None):
    QMessageBox.information(parent, "Ultra Lottery Helper", msg)

# ---------------- Plot Widget ----------------

class MatplotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

    def set_figure(self, fig):
        if self._canvas:
            self.layout().removeWidget(self._canvas)
            self._canvas.setParent(None)
        self._canvas = FigureCanvas(fig)
        self.layout().addWidget(self._canvas)
        self._canvas.draw()

# ---------------- Game Tab ----------------

class GameTab(QWidget):
    def __init__(self, game_key: str, parent=None):
        super().__init__(parent)
        self.game_key = game_key
        self.spec = GAMES[game_key]
        self.history_df = pd.DataFrame()
        self.portfolio_df = pd.DataFrame()
        self.plot_cache = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # --- row: reload
        row1 = QHBoxLayout()
        self.chk_online = QCheckBox("Fetch online (optional)")
        self.btn_reload = QPushButton("Reload history")
        self.btn_preset = QPushButton("Preset: Max Run")
        row1.addWidget(self.chk_online); row1.addStretch(1)
        row1.addWidget(self.btn_reload); row1.addWidget(self.btn_preset)
        root.addLayout(row1)

        self.lbl_log = QLabel("History not loaded."); root.addWidget(self.lbl_log)

        # --- settings
        root.addWidget(self._build_settings_group())

        # --- actions
        row2 = QHBoxLayout()
        self.btn_predict = QPushButton("Predict Portfolio")
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_png = QPushButton("Export PNG")
        row2.addWidget(self.btn_predict); row2.addStretch(1)
        row2.addWidget(self.btn_export_csv); row2.addWidget(self.btn_export_png)
        root.addLayout(row2)

        # table
        self.tbl = QTableWidget(0, 0)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        root.addWidget(self.tbl)

        # info
        self.lbl_warn = QLabel(""); self.lbl_warn.setStyleSheet("color:#b45309;")
        self.lbl_risk = QLabel(""); self.lbl_risk.setStyleSheet("color:#0369a1;")
        root.addWidget(self.lbl_warn); root.addWidget(self.lbl_risk)

        # plots
        rowp1 = QHBoxLayout()
        self.plt1 = MatplotWidget(); self.plt2 = MatplotWidget(); self.plt3 = MatplotWidget()
        rowp1.addWidget(self.plt1); rowp1.addWidget(self.plt2); rowp1.addWidget(self.plt3)
        root.addLayout(rowp1)
        rowp2 = QHBoxLayout()
        self.plt4 = MatplotWidget(); self.plt5 = MatplotWidget()
        rowp2.addWidget(self.plt4); rowp2.addWidget(self.plt5)
        root.addLayout(rowp2)

        # connect
        self.btn_reload.clicked.connect(self.on_reload)
        self.btn_preset.clicked.connect(self.on_preset)
        self.btn_predict.clicked.connect(self.on_predict)
        self.btn_export_csv.clicked.connect(self.on_export_csv)
        self.btn_export_png.clicked.connect(self.on_export_png)

    def _build_settings_group(self):
        g = QGroupBox("Settings")
        grid = QGridLayout(g)
        # iterations/topk/seed
        grid.addWidget(QLabel("Iterations"),0,0)
        self.spn_iter = QSpinBox(); self.spn_iter.setRange(1000,500000); self.spn_iter.setValue(50000); self.spn_iter.setSingleStep(1000)
        grid.addWidget(self.spn_iter,0,1)
        grid.addWidget(QLabel("Top-K"),0,2)
        self.spn_topk = QSpinBox(); self.spn_topk.setRange(10,1000); self.spn_topk.setValue(200); self.spn_topk.setSingleStep(10)
        grid.addWidget(self.spn_topk,0,3)
        grid.addWidget(QLabel("Seed"),0,4)
        self.spn_seed = QSpinBox(); self.spn_seed.setRange(0,10**9); self.spn_seed.setValue(42)
        grid.addWidget(self.spn_seed,0,5)
        return g

    def _collect_cfg(self) -> Config:
        return Config(
            iterations=self.spn_iter.value(),
            topk=self.spn_topk.value(),
            seed=self.spn_seed.value(),
            use_bma=True,
            use_ewma=True
        )

    # --- slots
    @Slot()
    def on_reload(self):
        try:
            use_online = self.chk_online.isChecked()
            df, log = _load_all_history(self.game_key, use_online=use_online)
            self.history_df = df
            self.lbl_log.setText(log)
        except Exception as e:
            error_box(str(e), self)

    @Slot()
    def on_preset(self):
        self.spn_iter.setValue(200000)
        self.spn_topk.setValue(500)
        self.spn_seed.setValue(777)

    @Slot()
    def on_predict(self):
        try:
            if len(self.history_df)==0:
                error_box("No history loaded.", self); return
            cfg = self._collect_cfg()
            cands, warning = generate_candidates(self.history_df, self.game_key, cfg, progress=None)
            mpb, spb = build_probs(self.history_df, self.game_key, cfg)
            port = dpp_select(cands, self.game_key, cfg)[:cfg.portfolio_size]
            mean_hit, risk = monte_carlo_risk(port, mpb, spb, self.spec, cfg)
            port_ev = apply_ev_rerank(self.game_key, cfg, port, mpb, spb)

            rows=[]; headers=[f"n{i+1}" for i in range(self.spec.main_pick)]
            if self.spec.sec_pick==1: headers+=["bonus"]
            elif self.spec.sec_pick==2: headers+=["e1","e2"]
            for m,s,sc,nev in port_ev[:6]:
                row=list(m)
                if self.spec.sec_pick==1: row+=[s]
                elif self.spec.sec_pick==2: row+=list(s)
                rows.append(row)
            self.portfolio_df=pd.DataFrame(rows,columns=headers)
            self._refresh_table()

            self.lbl_warn.setText(warning or "")
            self.lbl_risk.setText(f"MeanHits={mean_hit:.2f}, Std={risk:.2f}")

            # plots
            key=_quick_df_sig(self.history_df,self.game_key)
            cached=self.plot_cache.get(key)
            if cached:
                f1,f2,f3,f4,f5=cached
            else:
                f1=plot_frequency(self.history_df,self.game_key)
                f2=plot_recency(self.history_df,self.game_key)
                f3=plot_last_digit(self.history_df,self.game_key)
                f4=plot_pairs_heatmap(self.history_df,self.game_key)
                f5=plot_odd_even(self.history_df,self.game_key)
                self.plot_cache[key]=(f1,f2,f3,f4,f5)
            self.plt1.set_figure(f1); self.plt2.set_figure(f2); self.plt3.set_figure(f3)
            self.plt4.set_figure(f4); self.plt5.set_figure(f5)
        except Exception as e:
            error_box(f"Predict failed:\n{e}\n{traceback.format_exc()}", self)

    @Slot()
    def on_export_csv(self):
        if len(self.portfolio_df)==0: return
        path=export_six_to_csv(self.portfolio_df,self.game_key,prefix=f"{self.game_key.lower()}_six")
        info_box(f"CSV exported to {path}",self)

    @Slot()
    def on_export_png(self):
        if len(self.portfolio_df)==0: return
        path=export_six_to_png(self.portfolio_df,self.game_key,title=f"{self.game_key} — 6 Recommended Columns",prefix=f"{self.game_key.lower()}_six")
        info_box(f"PNG exported to {path}",self)

    def _refresh_table(self):
        df=self.portfolio_df
        self.tbl.setRowCount(len(df)); self.tbl.setColumnCount(len(df.columns))
        self.tbl.setHorizontalHeaderLabels(list(df.columns))
        for r in range(len(df)):
            for c in range(len(df.columns)):
                it=QTableWidgetItem(str(df.iat[r,c]))
                it.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(r,c,it)

# ---------------- Main Window ----------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultra Lottery Helper — Desktop")
        self.setMinimumSize(QSize(1000,700))
        icon_path = HERE.parent / "assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        tabs=QTabWidget()
        tabs.addTab(GameTab("TZOKER"),"TZOKER")
        tabs.addTab(GameTab("LOTTO"),"LOTTO")
        tabs.addTab(GameTab("EUROJACKPOT"),"EUROJACKPOT")
        self.setCentralWidget(tabs)

        bar=self.menuBar()
        m_help=bar.addMenu("&Help")
        act_about=QAction("About",self)
        act_about.triggered.connect(self._about)
        m_help.addAction(act_about)

    def _about(self):
        info_box("Ultra Lottery Helper — Native Desktop UI\nOffline, local.\nData stays on your PC.",self)

# ---------------- Entry ----------------

def main():
    app=QApplication(sys.argv)
    win=MainWindow(); win.show()
    return app.exec()

if __name__=="__main__":
    sys.exit(main())
