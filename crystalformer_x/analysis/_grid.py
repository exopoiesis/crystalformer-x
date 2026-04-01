"""Shared distance grid construction for void and percolation analysis.

Used by both :mod:`voronoi` and :mod:`percolation` to avoid duplicating
the expensive KD-tree + grid-sampling step.
"""

from itertools import product

import numpy as np
from scipy.spatial import cKDTree


def build_distance_grid(structure, grid_resolution: float = 0.2):
    """Return a 3-D grid of distance-to-nearest-atom values.

    Uses a 3x3x3 supercell of atom positions to correctly handle
    periodic boundary conditions, then samples the central cell on a
    regular grid and queries the nearest atom distance via a KD-tree.

    Parameters:
        structure: A ``pymatgen.Structure``.
        grid_resolution: Spacing of the sampling grid in angstrom.

    Returns:
        (distance_grid, grid_shape) where *distance_grid* is a 3-D
        numpy array of shape *grid_shape* = (na, nb, nc).
    """
    lattice = structure.lattice
    frac_coords = structure.frac_coords

    # 3x3x3 supercell atom positions in Cartesian
    offsets = np.array(list(product([-1, 0, 1], repeat=3)))  # (27, 3)
    sc_frac = np.concatenate(
        [frac_coords + off for off in offsets], axis=0
    )
    sc_cart = lattice.get_cartesian_coords(sc_frac)
    tree = cKDTree(sc_cart)

    # Sampling grid inside the central cell
    na = max(2, int(np.ceil(lattice.a / grid_resolution)))
    nb = max(2, int(np.ceil(lattice.b / grid_resolution)))
    nc = max(2, int(np.ceil(lattice.c / grid_resolution)))

    fa = np.linspace(0, 1, na, endpoint=False)
    fb = np.linspace(0, 1, nb, endpoint=False)
    fc = np.linspace(0, 1, nc, endpoint=False)
    grid_frac = np.stack(
        np.meshgrid(fa, fb, fc, indexing="ij"), axis=-1
    )  # (na, nb, nc, 3)

    grid_cart = lattice.get_cartesian_coords(
        grid_frac.reshape(-1, 3)
    )

    distances, _ = tree.query(grid_cart)
    distance_grid = distances.reshape(na, nb, nc)

    return distance_grid, (na, nb, nc)
