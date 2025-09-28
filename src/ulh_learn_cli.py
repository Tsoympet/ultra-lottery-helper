
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI for ULH learning loop.

Examples
--------
# Record a portfolio (space-separated numbers per line)
python -m ulh_learn_cli record-portfolio TZOKER "1 5 12 27 38" "3 14 22 33 41"

# Record an outcome
python -m ulh_learn_cli record-outcome TZOKER --main "3 14 22 33 41" --sec "5"

# Learn after draw and print metrics/state
python -m ulh_learn_cli learn TZOKER --k 100 --replay 2
"""
import argparse, json, sys
from typing import List
from ulh_learning import record_portfolio, record_outcome, learn_after_draw

def parse_combo(s: str) -> List[int]:
    return [int(x) for x in s.replace(",", " ").split()]

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    rp = sub.add_parser("record-portfolio")
    rp.add_argument("game")
    rp.add_argument("combos", nargs="+", help="e.g., \"1 5 12 27 38\"")
    rp.add_argument("--tag", default=None)

    ro = sub.add_parser("record-outcome")
    ro.add_argument("game")
    ro.add_argument("--main", required=True, help="e.g., \"1 5 12 27 38\"")
    ro.add_argument("--sec", default=None, help="e.g., \"2\" or \"2 7\" (if applicable)")
    ro.add_argument("--ts", default=None, help="ISO date/datetime, defaults to today UTC")

    le = sub.add_parser("learn")
    le.add_argument("game")
    le.add_argument("--k", type=int, default=None, help="evaluate only the top-K latest combos")
    le.add_argument("--replay", type=int, default=1, help="walk-forward replay rounds")

    args = ap.parse_args()

    if args.cmd == "record-portfolio":
        combos = [parse_combo(c) for c in args.combos]
        n = record_portfolio(args.game, combos, tag=args.tag)
        print(json.dumps({"saved_combos": n}, indent=2))

    elif args.cmd == "record-outcome":
        sec = parse_combo(args.sec) if args.sec else None
        n = record_outcome(args.game, parse_combo(args.main), sec, ts_draw=args.ts)
        print(json.dumps({"saved_outcomes": n}, indent=2))

    elif args.cmd == "learn":
        out = learn_after_draw(args.game, k_limit=args.k, self_replay_rounds=args.replay)
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
