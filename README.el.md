# Υπέρτατος Βοηθός Λαχείων (Ultra Lottery Helper)
[English](README.md) | [Ελληνικά](README.el.md)

Ο **Υπέρτατος Βοηθός Λαχείων** είναι εργαλείο **offline** για ανάλυση/παραγωγή στηλών για **ΤΖΟΚΕΡ / ΛΟΤΤΟ / EuroJackpot**.
Συνδυάζει **EWMA/BMA**, προσαρμοσμένα **luck/unluck**, **περιορισμούς**, **Gumbel Top‑k**, επιλογή χαρτοφυλακίου **DPP**, ρίσκο **Monte Carlo** και προαιρετικά **ML**. Ξεκινά με **Gradio** UI.

## Γρήγορη Εκκίνηση
- **Windows Installer**: Κατεβάστε `UltraLotteryHelperInstaller_X.Y.Z.exe` από Releases.
- **Φορητή**: `ultra_lottery_helper.exe` (προαιρετικά βάλτε φάκελο `data/` δίπλα).
- **Dev**: `pip install -r requirements.txt` → `python src/ultra_lottery_helper.py`

## Δεδομένα
`data/history/{tzoker,lotto,eurojackpot}`, εξαγωγές σε `exports/*`.

## Άδεια
MIT (`LICENSE.txt`). Παίξτε υπεύθυνα.
