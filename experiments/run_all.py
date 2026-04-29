"""Top-level orchestrator. Runs all sweeps with multi-seed and writes JSON to results_v2/raw."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make `kbm` and `runners` importable when executed as a script.
sys.path.insert(0, str(Path(__file__).parent))

from runners.equivalence import EquivConfig, run_sweep as run_equiv
from runners.spectral_bias import SpectralConfig, run_sweep as run_spectral
from runners.rank_vs_width import RankConfig, run_sweep as run_rank
from runners.phase_transition import PhaseConfig, run_sweep as run_phase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results_v2/raw")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--include", default="equiv,spectral,rank,phase",
                    help="comma list: equiv,spectral,rank,phase,fashion")
    ap.add_argument("--scale", default="medium",
                    help="small | medium | large; controls N and epochs.")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    include = set(args.include.split(","))

    if args.scale == "small":
        Ns = [10]
        n_epochs = 200
        spectral_N = 30
        spectral_epochs = 1500
    elif args.scale == "medium":
        Ns = [10, 30]
        n_epochs = 200
        spectral_N = 30
        spectral_epochs = 4000
    else:  # large
        Ns = [10, 30, 100]
        n_epochs = 300
        spectral_N = 50
        spectral_epochs = 5000

    if "equiv" in include:
        for N in Ns:
            for depth in [1, 2, 3]:
                cfg = EquivConfig(N=N, depth=depth, n_epochs=n_epochs)
                out_path = out_root / f"equivalence_N{N}_L{depth}.json"
                print(f"[equiv] N={N} L={depth} -> {out_path}", flush=True)
                run_equiv(cfg, seeds, out_path, device=args.device)

    if "spectral" in include:
        cfg = SpectralConfig(N=spectral_N, n_epochs=spectral_epochs, depth=1)
        out_path = out_root / f"spectral_bias_N{spectral_N}_L1.json"
        print(f"[spectral] -> {out_path}", flush=True)
        run_spectral(cfg, seeds, out_path, device=args.device)
        cfg2 = SpectralConfig(N=spectral_N, n_epochs=spectral_epochs, depth=2)
        out_path2 = out_root / f"spectral_bias_N{spectral_N}_L2.json"
        print(f"[spectral] -> {out_path2}", flush=True)
        run_spectral(cfg2, seeds, out_path2, device=args.device)

    if "rank" in include:
        cfg = RankConfig(n_epochs=n_epochs * 4)
        out_path = out_root / "rank_vs_width.json"
        print(f"[rank] -> {out_path}", flush=True)
        run_rank(cfg, seeds, out_path, device=args.device)

    if "phase" in include:
        cfg = PhaseConfig(n_epochs=n_epochs * 4)
        out_path = out_root / "phase_transition.json"
        print(f"[phase] -> {out_path}", flush=True)
        run_phase(cfg, seeds, out_path, device=args.device)

    if "fashion" in include:
        from runners.fashion_mnist import FashionConfig, run_sweep as run_fashion
        cfg = FashionConfig(n_epochs=n_epochs * 4)
        out_path = out_root / "fashion_mnist.json"
        print(f"[fashion] -> {out_path}", flush=True)
        run_fashion(cfg, seeds, out_path, device=args.device)

    print("done.")


if __name__ == "__main__":
    main()
