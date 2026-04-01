"""Shared CSV I/O for analysis and screening CLI tools."""

import csv
import sys
from ast import literal_eval

from pymatgen.core import Structure


def load_structures(csv_path: str) -> list[tuple[int, Structure]]:
    """Load structures from a CSV with a ``cif`` column.

    The ``cif`` column must contain ``Structure.as_dict()`` string
    representations (the format produced by ``scripts/awl2struct.py``).

    Returns a list of (row_index, Structure) tuples.  Rows that fail to
    parse are silently skipped with a warning on stderr.
    """
    structures = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "cif" not in (reader.fieldnames or []):
            print("ERROR: CSV must have a 'cif' column", file=sys.stderr)
            sys.exit(1)
        for i, row in enumerate(reader):
            try:
                s = Structure.from_dict(literal_eval(row["cif"]))
                structures.append((i, s))
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARNING: row {i} skipped ({exc})",
                    file=sys.stderr,
                )
    return structures
