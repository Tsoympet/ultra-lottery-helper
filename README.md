# Ultra Lottery Helper — Offline Suite (v6.3.0)

Offline-first εργαλείο ανάλυσης ιστορικού λαχείων (TZOKER, LOTTO, EUROJACKPOT) με:
- EWMA + BMA + προσαρμοστικά Luck/Unluck
- (Προαιρετικά) ML ensemble: RF / LightGBM / XGBoost / SVM / Prophet
- Gumbel Top-k sampling με constraints
- DPP επιλογή χαρτοφυλακίου + Monte Carlo ρίσκο
- EV (cost-aware) re-rank (προαιρετικά)
- Gradio UI με **plot caching** και **debounce-style feedback** σε βαριά sliders

> **Disclaimer:** Οι κληρώσεις είναι τυχαίες. Τα αποτελέσματα είναι πιθανοτικά και δεν εγγυώνται κέρδη. Παίξτε υπεύθυνα.

## Γρήγορη εκκίνηση (local)
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python src/ultra_lottery_helper.py
```

## Colab
- Δες `examples/colab_setup_snippet.txt` για έτοιμες εντολές.
- Τρέξε το `ultra_lottery_helper.py` — paths είναι Colab-friendly (`/content/...`).

## Φάκελοι Δεδομένων
Βάλε τα ιστορικά αρχεία (CSV/XLS/XLSX) στους φακέλους:
```
data/history/tzoker/       # cols: n1..n5, joker [, date]
data/history/lotto/        # cols: n1..n6       [, date]
data/history/eurojackpot/  # cols: n1..n5, e1, e2 [, date]
```
Στο UI, μπορείς προαιρετικά να τικάρεις **Fetch online history** για γρήγορο refresh.

## Export
Τα αποτελέσματα (CSV/PNG) αποθηκεύονται σε:
```
exports/tzoker/
exports/lotto/
exports/eurojackpot/
```

## EV (προαιρετικά)
Πέρασε prize tiers JSON (παράδειγμα: `examples/prize_tiers_example.json`), όρισε ticket price, και ενεργοποίησε **EV re-rank**.

## Troubleshooting
- **Prophet**: απαιτεί `cmdstanpy` στο παρασκήνιο. Αν δεν θες Prophet, άφησέ τον απενεργοποιημένο (default).
- **XGBoost/LightGBM**: είναι προαιρετικά. Αν λείπουν, το script συνεχίζει χωρίς αυτά.
- **Υψηλή μνήμη/CPU**: μείωσε `iterations`, `topk`, `monte_sims`.

Καλή πλώρη! ⚓🎲
