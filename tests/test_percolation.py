"""Tests for crystalformer_x.analysis.percolation -- PercolationAnalyzer.

Test structures are designed so that percolation answers are
geometrically obvious and verifiable by inspection.
"""

import pytest
import numpy as np

pytest.importorskip("pymatgen", reason="pymatgen required for analysis tests")

from pymatgen.core import Structure, Lattice

from crystalformer_x.analysis import PercolationAnalyzer, PercolationResult


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #

@pytest.fixture
def open_cubic():
    """Single atom in a large cubic cell -- almost entirely void."""
    return Structure(Lattice.cubic(8.0), ["Si"], [[0.0, 0.0, 0.0]])


@pytest.fixture
def dense_nacl():
    """NaCl (Fm-3m), dense packing."""
    return Structure.from_spacegroup(
        225, Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0, 0]]
    )


@pytest.fixture
def layered_mos2():
    """MoS2-like with large interlayer gap along c."""
    return Structure(
        Lattice.from_parameters(3.2, 3.2, 20.0, 90, 90, 120),
        ["Mo", "S", "S"],
        [[0.0, 0.0, 0.0], [0.333, 0.667, 0.08], [0.333, 0.667, 0.92]],
    )


@pytest.fixture
def tube_along_c():
    """Atoms arranged on a ring in the ab plane, leaving a channel along c.

    8 atoms at radius ~ 3 A from the cell center in a 10x10x4 cell.
    The center (5, 5, z) is far from all atoms -> clear 1D channel in c.
    """
    species = ["Si"] * 8
    coords_cart = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        x = 0.5 + 0.3 * np.cos(angle)  # fractional in a 10 A cell
        y = 0.5 + 0.3 * np.sin(angle)
        coords_cart.append([x, y, 0.25 * (i % 2)])  # stagger in z
    return Structure(
        Lattice.from_parameters(10.0, 10.0, 4.0, 90, 90, 90),
        species,
        coords_cart,
    )


# ------------------------------------------------------------------ #
#  PercolationResult contract                                         #
# ------------------------------------------------------------------ #

class TestPercolationResultContract:
    def test_has_required_fields(self, open_cubic):
        result = PercolationAnalyzer().analyze(open_cubic)
        assert isinstance(result, PercolationResult)
        for attr in (
            "percolates_a",
            "percolates_b",
            "percolates_c",
            "percolation_dimensionality",
            "min_bottleneck",
        ):
            assert hasattr(result, attr), f"missing {attr}"

    def test_dimensionality_matches_bools(self, open_cubic):
        r = PercolationAnalyzer(r_probe=1.0).analyze(open_cubic)
        expected = sum([r.percolates_a, r.percolates_b, r.percolates_c])
        assert r.percolation_dimensionality == expected


# ------------------------------------------------------------------ #
#  Open cell -- percolates everywhere                                 #
# ------------------------------------------------------------------ #

class TestOpenCellPercolation:
    def test_percolates_3d(self, open_cubic):
        """Single atom in 8 A cell with 1 A probe -> percolates everywhere."""
        r = PercolationAnalyzer(r_probe=1.0).analyze(open_cubic)
        assert r.percolates_a is True
        assert r.percolates_b is True
        assert r.percolates_c is True
        assert r.percolation_dimensionality == 3

    def test_huge_probe_no_percolation(self, open_cubic):
        """Probe larger than max void -> no percolation."""
        r = PercolationAnalyzer(r_probe=10.0).analyze(open_cubic)
        assert r.percolation_dimensionality == 0

    def test_bottleneck_positive_when_percolating(self, open_cubic):
        r = PercolationAnalyzer(r_probe=1.0).analyze(open_cubic)
        assert r.min_bottleneck is not None
        assert r.min_bottleneck > 1.0

    def test_bottleneck_none_when_not_percolating(self, open_cubic):
        r = PercolationAnalyzer(r_probe=10.0).analyze(open_cubic)
        assert r.min_bottleneck is None


# ------------------------------------------------------------------ #
#  Dense cell -- blocked                                              #
# ------------------------------------------------------------------ #

class TestDenseCellPercolation:
    def test_nacl_large_probe_blocked(self, dense_nacl):
        """NaCl with probe r=2.0 A -- not enough space to percolate."""
        r = PercolationAnalyzer(r_probe=2.0).analyze(dense_nacl)
        assert r.percolation_dimensionality == 0


# ------------------------------------------------------------------ #
#  Layered structure                                                  #
# ------------------------------------------------------------------ #

class TestLayeredPercolation:
    def test_layered_percolates(self, layered_mos2):
        """Large interlayer gap should allow percolation."""
        r = PercolationAnalyzer(r_probe=1.0).analyze(layered_mos2)
        assert r.percolation_dimensionality >= 1

    def test_layered_percolates_in_plane(self, layered_mos2):
        """Interlayer void extends in the ab plane -> a and b should percolate."""
        r = PercolationAnalyzer(r_probe=1.0).analyze(layered_mos2)
        assert r.percolates_a is True
        assert r.percolates_b is True


# ------------------------------------------------------------------ #
#  Tube / channel structure                                           #
# ------------------------------------------------------------------ #

class TestTubePercolation:
    def test_tube_percolates_along_c(self, tube_along_c):
        """Ring of atoms leaves a channel along c."""
        r = PercolationAnalyzer(r_probe=0.5).analyze(tube_along_c)
        assert r.percolates_c is True


# ------------------------------------------------------------------ #
#  Probe sensitivity                                                  #
# ------------------------------------------------------------------ #

class TestProbeSensitivity:
    def test_smaller_probe_at_least_as_permeable(self, dense_nacl):
        """Shrinking the probe can only open more channels, never close them."""
        r_small = PercolationAnalyzer(r_probe=0.5).analyze(dense_nacl)
        r_large = PercolationAnalyzer(r_probe=2.0).analyze(dense_nacl)
        assert r_small.percolation_dimensionality >= r_large.percolation_dimensionality
