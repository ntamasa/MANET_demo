"""Command-line runner (PLAN.md §12).

    python cli.py --size 40 40 --seed 42 --algo DADENCE [--no-dealloc] [--gif out.gif] [--csv hist.csv]

Prints the run summary; exit code 0 on success (100% coverage), 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys

from manet.algorithms import ALGORITHM_NAMES
from manet.render import make_gif
from manet.sim import run
from manet.summary import format_summary, history_frame, make_algorithm, summary_dict
from manet.world import generate_world


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MANET coverage demo (CADENCE / DADENCE / ISDA)")
    p.add_argument("--size", nargs=2, type=int, default=[40, 40], metavar=("H", "W"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--algo", type=str.upper, choices=ALGORITHM_NAMES, default="CADENCE")
    p.add_argument("--no-dealloc", dest="dealloc", action="store_false")
    p.add_argument("--no-early-spawn", dest="early_spawn", action="store_false",
                   help="DADENCE only")
    p.add_argument("--n-max", type=int, default=None, help="0/omitted = auto (world Nmax)")
    p.add_argument("--t-max", type=int, default=None, help="0/omitted = auto (100*max(H,W))")
    p.add_argument("--gif", metavar="PATH", default=None)
    p.add_argument("--every", type=int, default=1, help="GIF: keep every N-th frame")
    p.add_argument("--csv", metavar="PATH", default=None, help="write step history as CSV")
    p.add_argument("--pillar-prob", type=float, default=0.3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    h, w = args.size
    world = generate_world(h, w, args.seed, pillar_prob=args.pillar_prob)
    algo = make_algorithm(args.algo, world, seed=args.seed, dealloc=args.dealloc, n_max=args.n_max,
                          t_max=args.t_max, early_spawn=args.early_spawn)
    snaps = []
    for snap in run(algo):
        if args.gif:
            snaps.append(snap)
    d = summary_dict(args.algo, world, args.seed, algo)
    print(format_summary(d))
    if args.csv:
        history_frame(algo.metrics).to_csv(args.csv)
        print(f"CSV: {args.csv}")
    if args.gif:
        make_gif(world, snaps, args.gif, every=args.every)
        print(f"GIF: {args.gif}")
    return 0 if d["success"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles: avoid UnicodeEncodeError on ✔/✘/ő
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
