"""CLI entry point: ``python -m crystalformer_x.analysis``.

Usage examples::

    # Full analysis (Voronoi + percolation)
    python -m crystalformer_x.analysis structures.csv --r-probe 0.4 --check-percolation

    # Voronoi only (faster)
    python -m crystalformer_x.analysis structures.csv --voronoi-only

    # Custom output path
    python -m crystalformer_x.analysis structures.csv -o results.csv

Input CSV must contain a ``cif`` column with pymatgen ``Structure.as_dict()``
string representations (the format produced by ``scripts/awl2struct.py``).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from crystalformer_x.analysis._io import load_structures
from crystalformer_x.analysis.percolation import PercolationAnalyzer
from crystalformer_x.analysis.voronoi import VoronoiAnalyzer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m crystalformer_x.analysis",
        description="Structural analysis of generated crystal structures.",
    )
    parser.add_argument(
        "input", help="Path to CSV file with a 'cif' column."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <input>_analysis.csv).",
    )
    parser.add_argument(
        "--r-probe",
        type=float,
        default=0.4,
        help="Probe radius in angstrom (default 0.4).",
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=0.2,
        help="Grid spacing in angstrom (default 0.2).",
    )
    parser.add_argument(
        "--voronoi-only",
        action="store_true",
        help="Skip percolation analysis (faster).",
    )
    parser.add_argument(
        "--check-percolation",
        action="store_true",
        help="Run percolation analysis (default if --voronoi-only not set).",
    )

    args = parser.parse_args(argv)

    do_percolation = not args.voronoi_only
    if args.check_percolation:
        do_percolation = True

    output_path = args.output
    if output_path is None:
        stem = Path(args.input).stem
        output_path = str(Path(args.input).with_name(f"{stem}_analysis.csv"))

    # Load
    print(f"Loading structures from {args.input} ...", file=sys.stderr)
    entries = load_structures(args.input)
    print(f"  {len(entries)} structures loaded.", file=sys.stderr)

    if not entries:
        print("No structures to analyze.", file=sys.stderr)
        sys.exit(0)

    # Analyze
    va = VoronoiAnalyzer(
        r_probe=args.r_probe, grid_resolution=args.grid_resolution
    )
    pa = (
        PercolationAnalyzer(
            r_probe=args.r_probe,
            grid_resolution=max(args.grid_resolution, 0.3),
        )
        if do_percolation
        else None
    )

    direction_names = {0: "a", 1: "b", 2: "c"}

    results = []
    t0 = time.time()
    for idx, (row_i, struct) in enumerate(entries):
        vr = va.analyze(struct)
        row = {
            "row": row_i,
            "n_atoms": len(struct),
            "max_void_radius": f"{vr.max_void_radius:.3f}",
            "void_fraction": f"{vr.void_fraction:.4f}",
            "layeredness_score": f"{vr.layeredness_score:.4f}",
            "interlayer_spacing": (
                f"{vr.interlayer_spacing:.2f}"
                if vr.interlayer_spacing is not None
                else ""
            ),
            "stacking_direction": direction_names[vr.stacking_direction],
        }

        if pa is not None:
            pr = pa.analyze(struct)
            row.update(
                {
                    "percolates_a": pr.percolates_a,
                    "percolates_b": pr.percolates_b,
                    "percolates_c": pr.percolates_c,
                    "percolation_dim": pr.percolation_dimensionality,
                    "min_bottleneck": (
                        f"{pr.min_bottleneck:.3f}"
                        if pr.min_bottleneck is not None
                        else ""
                    ),
                }
            )

        results.append(row)

        if (idx + 1) % 100 == 0 or idx + 1 == len(entries):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{idx + 1}/{len(entries)}] "
                f"{rate:.1f} structs/sec",
                file=sys.stderr,
            )

    # Write
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {output_path}", file=sys.stderr)

    # Summary
    if do_percolation:
        n_perc = sum(1 for r in results if int(r["percolation_dim"]) > 0)
        print(
            f"  {n_perc}/{len(results)} structures have percolating channels.",
            file=sys.stderr,
        )
    n_layered = sum(
        1 for r in results if float(r["layeredness_score"]) > 0.7
    )
    print(
        f"  {n_layered}/{len(results)} structures are layered "
        f"(score > 0.7).",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
