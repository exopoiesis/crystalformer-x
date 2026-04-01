"""CLI entry point: ``python -m crystalformer_x.screen``.

Usage examples::

    # Screen with built-in filters
    python -m crystalformer_x.screen structures.csv \\
        --filters validity,voronoi,percolation \\
        --r-probe 0.4 --percolation-dim 1

    # Only keep Fe-S structures with large voids
    python -m crystalformer_x.screen structures.csv \\
        --filters validity,composition,voronoi \\
        --elements Fe S --r-min 1.0

Input CSV must have a ``cif`` column (pymatgen Structure.as_dict() format).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from crystalformer_x.analysis._io import load_structures
from crystalformer_x.screen.filters import (
    CompositionFilter,
    DensityFilter,
    PercolationFilter,
    ValidityFilter,
    VoronoiFilter,
)
from crystalformer_x.screen.pipeline import ScreeningPipeline


def _build_filters(args) -> list:
    """Build filter list from CLI arguments."""
    filter_names = [f.strip() for f in args.filters.split(",")]
    filters = []

    for name in filter_names:
        if name == "validity":
            filters.append(ValidityFilter(min_dist=args.min_dist))
        elif name == "voronoi":
            filters.append(VoronoiFilter(r_min=args.r_min))
        elif name == "percolation":
            filters.append(
                PercolationFilter(
                    min_dim=args.percolation_dim, r_probe=args.r_probe
                )
            )
        elif name == "composition":
            if not args.elements:
                print(
                    "Error: composition filter requires --elements",
                    file=sys.stderr,
                )
                sys.exit(2)
            filters.append(CompositionFilter(elements=args.elements))
        elif name == "density":
            filters.append(
                DensityFilter(
                    min_density=args.min_density,
                    max_density=args.max_density,
                )
            )
        else:
            print(
                f"Error: unknown filter {name!r}. "
                f"Available: validity, voronoi, percolation, composition, density",
                file=sys.stderr,
            )
            sys.exit(2)

    return filters


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m crystalformer_x.screen",
        description="Screen generated crystal structures through a filter pipeline.",
        epilog=(
            "Required combinations:\n"
            "  --filters composition  requires --elements\n\n"
            "Examples:\n"
            "  %(prog)s structures.csv --filters validity,voronoi\n"
            "  %(prog)s structures.csv --filters validity,composition "
            "--elements Fe S\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to CSV with a 'cif' column.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path (default: <input>_screened.csv).",
    )
    parser.add_argument(
        "--filters", default="validity,voronoi",
        help="Comma-separated filter names: validity, voronoi, percolation, "
        "composition, density (default: validity,voronoi).",
    )

    # Filter-specific parameters
    parser.add_argument("--min-dist", type=float, default=0.5,
                        help="ValidityFilter: min interatomic distance (default 0.5 A).")
    parser.add_argument("--r-min", type=float, default=1.0,
                        help="VoronoiFilter: min void radius (default 1.0 A).")
    parser.add_argument("--r-probe", type=float, default=0.4,
                        help="PercolationFilter: probe radius (default 0.4 A).")
    parser.add_argument("--percolation-dim", type=int, default=1,
                        help="PercolationFilter: min percolation dimensionality (default 1).")
    parser.add_argument("--elements", nargs="+", default=None,
                        help="CompositionFilter: required elements, e.g. --elements Fe S.")
    parser.add_argument("--min-density", type=float, default=None,
                        help="DensityFilter: minimum density (g/cm3).")
    parser.add_argument("--max-density", type=float, default=None,
                        help="DensityFilter: maximum density (g/cm3).")

    args = parser.parse_args(argv)

    output_path = args.output
    if output_path is None:
        stem = Path(args.input).stem
        output_path = str(Path(args.input).with_name(f"{stem}_screened.csv"))

    # Load
    print(f"Loading structures from {args.input} ...", file=sys.stderr)
    entries = load_structures(args.input)
    print(f"  {len(entries)} structures loaded.", file=sys.stderr)

    if not entries:
        print("No structures to screen.", file=sys.stderr)
        sys.exit(0)

    indices, structures = zip(*entries)
    structures = list(structures)

    # Build pipeline
    filters = _build_filters(args)
    pipeline = ScreeningPipeline(filters)

    # Run
    t0 = time.time()
    result = pipeline.run(structures)
    elapsed = time.time() - t0

    # Print summary
    print(f"\n{result.summary()}", file=sys.stderr)
    print(f"\n  Time: {elapsed:.1f}s ({len(structures)/elapsed:.1f} structs/sec)",
          file=sys.stderr)

    # Write survivors
    # Re-read original CSV to preserve all columns
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
        fieldnames = reader.fieldnames or []

    # Map pipeline survivor indices back to original CSV row indices
    survivor_csv_indices = {indices[si] for si in result.survivor_indices}

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(all_rows):
            if i in survivor_csv_indices:
                writer.writerow(row)

    print(f"  {len(result.survivors)} structures written to {output_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
