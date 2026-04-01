"""Percolation analysis of crystal structures.

Determines whether a probe of given radius can traverse the unit cell
through connected void space, checking each lattice direction
independently.

Algorithm: build a 3-D accessibility grid, tile it 2x2x2 (to expose
periodic connections without explicit PBC bookkeeping), label connected
components, then check whether any component in the original cell
connects to its periodic image.

Dependencies: pymatgen, scipy, numpy -- install via
``pip install crystalformer-x``.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import label as ndimage_label

from crystalformer_x.analysis._grid import build_distance_grid


@dataclass
class PercolationResult:
    """Result of percolation analysis.

    Attributes:
        percolates_a: Whether a continuous channel exists along *a*.
        percolates_b: Whether a continuous channel exists along *b*.
        percolates_c: Whether a continuous channel exists along *c*.
        percolation_dimensionality: Number of percolating directions (0-3).
        min_bottleneck: Minimum distance-to-nearest-atom in the
            percolating region (angstrom).  ``None`` when no direction
            percolates.
    """

    percolates_a: bool
    percolates_b: bool
    percolates_c: bool
    percolation_dimensionality: int
    min_bottleneck: Optional[float]


# 6-connectivity structuring element for 3-D labeling
_STRUCT_6 = np.zeros((3, 3, 3), dtype=bool)
_STRUCT_6[1, 1, 0] = True
_STRUCT_6[1, 1, 2] = True
_STRUCT_6[1, 0, 1] = True
_STRUCT_6[1, 2, 1] = True
_STRUCT_6[0, 1, 1] = True
_STRUCT_6[2, 1, 1] = True
_STRUCT_6[1, 1, 1] = True  # center


class PercolationAnalyzer:
    """Check whether void channels percolate through a crystal.

    Parameters:
        r_probe: Probe radius in angstrom (default 0.4, ~ H+ radius).
        grid_resolution: Spacing of the sampling grid in angstrom.
    """

    def __init__(self, r_probe: float = 0.4, grid_resolution: float = 0.3):
        self.r_probe = r_probe
        self.grid_resolution = grid_resolution

    def analyze(self, structure) -> PercolationResult:
        """Analyze a single pymatgen Structure."""
        distance_grid, grid_shape = build_distance_grid(
            structure, self.grid_resolution
        )
        accessible = distance_grid > self.r_probe

        perc_flags, min_btl = self._check_percolation(
            accessible, distance_grid, grid_shape
        )

        dim = sum(perc_flags)
        return PercolationResult(
            percolates_a=perc_flags[0],
            percolates_b=perc_flags[1],
            percolates_c=perc_flags[2],
            percolation_dimensionality=dim,
            min_bottleneck=min_btl,
        )

    @staticmethod
    def _check_percolation(accessible, distance_grid, grid_shape):
        """Doubled-grid percolation check.

        Tile the accessibility mask 2x2x2, label connected components
        with 6-connectivity, then test whether any component in the
        first cell also appears in the shifted copy along each axis.

        Returns:
            (list[bool] for a/b/c, Optional[float] min_bottleneck)
        """
        na, nb, nc = grid_shape

        if not np.any(accessible):
            return [False, False, False], None

        # 2x2x2 tile
        big = np.tile(accessible, (2, 2, 2))
        labels, _ = ndimage_label(big, structure=_STRUCT_6)

        # Distance grid tiled the same way (for bottleneck)
        big_dist = np.tile(distance_grid, (2, 2, 2))

        # Original cell labels
        orig = labels[:na, :nb, :nc]

        perc = [False, False, False]
        dims = [na, nb, nc]
        percolating_labels = set()

        for d in range(3):
            slc_shift = [slice(0, na), slice(0, nb), slice(0, nc)]
            slc_shift[d] = slice(dims[d], 2 * dims[d])

            shifted = labels[tuple(slc_shift)]

            # Both cells must be accessible (label > 0)
            mask = (orig > 0) & (shifted > 0)
            if not np.any(mask):
                continue

            matching = orig[mask] == shifted[mask]
            if np.any(matching):
                perc[d] = True
                percolating_labels.update(orig[mask][matching].tolist())

        # Min bottleneck across all percolating components
        if percolating_labels:
            perc_mask = np.isin(labels, list(percolating_labels))
            min_btl = float(np.min(big_dist[perc_mask]))
        else:
            min_btl = None

        return perc, min_btl
