"""Voronoi-based structural analysis of crystal structures.

Provides void characterization (max void radius, void fraction)
and layeredness analysis (gap-based score, interlayer spacing).

Dependencies: pymatgen, scipy, numpy -- install via
``pip install crystalformer-x``.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from crystalformer_x.analysis._grid import build_distance_grid


@dataclass
class VoronoiResult:
    """Result of Voronoi structural analysis.

    Attributes:
        max_void_radius: Radius of the largest empty sphere (angstrom).
        void_fraction: Fraction of unit cell volume accessible to a probe
            of radius ``r_probe`` (0-1).
        layeredness_score: Largest fractional gap along any lattice
            direction (0 = uniform, 1 = all atoms in one plane).
        interlayer_spacing: Gap between atomic layers in the stacking
            direction (angstrom). ``None`` when structure has fewer than
            2 atoms.
        stacking_direction: Lattice direction with the largest gap
            (0 = a, 1 = b, 2 = c).
    """

    max_void_radius: float
    void_fraction: float
    layeredness_score: float
    interlayer_spacing: Optional[float]
    stacking_direction: int


class VoronoiAnalyzer:
    """Analyze void structure and layeredness of crystals.

    Parameters:
        r_probe: Probe radius in angstrom (default 0.4, approximate H+
            radius).
        grid_resolution: Spacing of the sampling grid in angstrom.
            Smaller values give more accurate ``max_void_radius`` and
            ``void_fraction`` at higher computational cost.
    """

    def __init__(self, r_probe: float = 0.4, grid_resolution: float = 0.2):
        self.r_probe = r_probe
        self.grid_resolution = grid_resolution

    # -------------------------------------------------------------- #
    #  public API                                                      #
    # -------------------------------------------------------------- #

    def analyze(self, structure) -> VoronoiResult:
        """Analyze a single pymatgen Structure."""
        distance_grid, grid_shape = build_distance_grid(
            structure, self.grid_resolution
        )
        max_void = float(np.max(distance_grid))
        void_frac = float(np.mean(distance_grid > self.r_probe))
        score, direction, spacing = self._compute_layeredness(structure)

        return VoronoiResult(
            max_void_radius=max_void,
            void_fraction=void_frac,
            layeredness_score=score,
            interlayer_spacing=spacing,
            stacking_direction=direction,
        )

    def analyze_batch(self, structures) -> list[VoronoiResult]:
        """Analyze a list of pymatgen Structures."""
        return [self.analyze(s) for s in structures]

    @staticmethod
    def _compute_layeredness(structure):
        """Compute gap-based layeredness score.

        For each lattice direction d the fractional coordinates are
        sorted and the largest gap (including the periodic wrap-around)
        is recorded.  The score is the maximum gap across all three
        directions.

        Returns:
            (score, direction_index, interlayer_spacing_angstrom)
        """
        n_atoms = len(structure)
        if n_atoms < 2:
            return 0.0, 0, None

        frac = structure.frac_coords % 1.0  # ensure [0, 1)
        lengths = [
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
        ]

        best_gap = 0.0
        best_dir = 0

        for d in range(3):
            sorted_c = np.sort(frac[:, d])
            # Gaps between consecutive atoms
            gaps = np.diff(sorted_c)
            # Wrap-around gap
            wrap = 1.0 - sorted_c[-1] + sorted_c[0]
            max_gap = float(max(np.max(gaps), wrap))
            if max_gap > best_gap:
                best_gap = max_gap
                best_dir = d

        spacing = best_gap * lengths[best_dir]
        return best_gap, best_dir, spacing
