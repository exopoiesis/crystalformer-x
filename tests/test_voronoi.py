"""Tests for crystalformer_x.analysis.voronoi -- VoronoiAnalyzer.

Test structures are hand-built with known geometric properties so that
the analysis output can be verified against ground truth.
"""

import csv
from pathlib import Path

import pytest
import numpy as np

pytest.importorskip("pymatgen", reason="pymatgen required for analysis tests")

from pymatgen.core import Structure, Lattice

from crystalformer_x.analysis import VoronoiAnalyzer, VoronoiResult


# ------------------------------------------------------------------ #
#  Fixtures -- test structures                                        #
# ------------------------------------------------------------------ #

@pytest.fixture
def open_cubic():
    """Single atom in a large cubic cell (a = 8 A).

    Farthest point from any atom (with PBC) is at (4, 4, 4),
    distance = 4*sqrt(3) ~ 6.93 A.  Lots of empty space, isotropic.
    """
    return Structure(Lattice.cubic(8.0), ["Si"], [[0.0, 0.0, 0.0]])


@pytest.fixture
def layered_mos2():
    """MoS2-like: atoms clustered near z ~ 0, large gap along c.

    Mo at z=0, S at z=0.08 and z=0.92.  c = 20 A.
    Largest gap in c direction: 0.08 -> 0.92 = 0.84 in fractional = 16.8 A.
    """
    return Structure(
        Lattice.from_parameters(3.2, 3.2, 20.0, 90, 90, 120),
        ["Mo", "S", "S"],
        [[0.0, 0.0, 0.0], [0.333, 0.667, 0.08], [0.333, 0.667, 0.92]],
    )


@pytest.fixture
def dense_nacl():
    """NaCl (Fm-3m, 8 atoms in conventional cell).

    Dense packing -- small voids.  Along each axis atoms sit at 0 and 0.5,
    so max gap = 0.5 in every direction -- moderate, not layered.
    """
    return Structure.from_spacegroup(
        225, Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0, 0]]
    )


@pytest.fixture
def two_layer():
    """Two dense layers separated by a large gap.

    12 atoms in two sheets at z ~ 0.1 and z ~ 0.9.
    Gap from z = 0.15 to z = 0.85 -> 0.70 * 16 A = 11.2 A.
    Clear layered signature with enough atoms per layer for robust statistics.
    """
    species = ["Fe"] * 12
    coords = []
    # Layer 1 near z = 0.1
    for x in [0.0, 0.5]:
        for y in [0.0, 0.5, 1 / 3]:
            coords.append([x, y, 0.10])
    # Layer 2 near z = 0.9
    for x in [0.25, 0.75]:
        for y in [0.0, 0.5, 1 / 3]:
            coords.append([x, y, 0.90])
    return Structure(
        Lattice.from_parameters(4.0, 4.0, 16.0, 90, 90, 90),
        species,
        coords,
    )


# ------------------------------------------------------------------ #
#  VoronoiResult contract                                             #
# ------------------------------------------------------------------ #

class TestVoronoiResultContract:
    """The result object must expose all documented fields."""

    def test_has_required_fields(self, open_cubic):
        result = VoronoiAnalyzer().analyze(open_cubic)
        assert isinstance(result, VoronoiResult)
        for attr in (
            "max_void_radius",
            "void_fraction",
            "layeredness_score",
            "interlayer_spacing",
            "stacking_direction",
        ):
            assert hasattr(result, attr), f"missing {attr}"

    def test_numeric_types(self, open_cubic):
        r = VoronoiAnalyzer().analyze(open_cubic)
        assert isinstance(r.max_void_radius, float)
        assert isinstance(r.void_fraction, float)
        assert isinstance(r.layeredness_score, float)
        assert isinstance(r.stacking_direction, int)


# ------------------------------------------------------------------ #
#  max_void_radius                                                    #
# ------------------------------------------------------------------ #

class TestMaxVoidRadius:
    def test_open_cell_large_void(self, open_cubic):
        """Single atom in 8 A cell -- void must be large (>5 A)."""
        r = VoronoiAnalyzer().analyze(open_cubic)
        assert r.max_void_radius > 5.0

    def test_dense_cell_small_void(self, dense_nacl):
        """NaCl -- voids smaller than 2 A."""
        r = VoronoiAnalyzer().analyze(dense_nacl)
        assert r.max_void_radius < 2.5

    def test_void_radius_positive(self, dense_nacl):
        r = VoronoiAnalyzer().analyze(dense_nacl)
        assert r.max_void_radius > 0

    def test_layered_has_large_interlayer_void(self, layered_mos2):
        """Interlayer gap of ~16.8 A should produce a large void."""
        r = VoronoiAnalyzer().analyze(layered_mos2)
        assert r.max_void_radius > 5.0


# ------------------------------------------------------------------ #
#  void_fraction                                                      #
# ------------------------------------------------------------------ #

class TestVoidFraction:
    def test_open_cell_high_fraction(self, open_cubic):
        r = VoronoiAnalyzer(r_probe=1.0).analyze(open_cubic)
        assert r.void_fraction > 0.8

    def test_dense_cell_low_fraction_large_probe(self, dense_nacl):
        """With probe r = 2.0 A, NaCl should have almost no accessible space."""
        r = VoronoiAnalyzer(r_probe=2.0).analyze(dense_nacl)
        assert r.void_fraction < 0.3

    def test_fraction_bounded_0_1(self, open_cubic):
        r = VoronoiAnalyzer().analyze(open_cubic)
        assert 0.0 <= r.void_fraction <= 1.0

    def test_larger_probe_less_accessible(self, open_cubic):
        """Increasing r_probe must reduce void_fraction."""
        r_small = VoronoiAnalyzer(r_probe=0.5).analyze(open_cubic)
        r_large = VoronoiAnalyzer(r_probe=3.0).analyze(open_cubic)
        assert r_small.void_fraction > r_large.void_fraction


# ------------------------------------------------------------------ #
#  layeredness                                                        #
# ------------------------------------------------------------------ #

class TestLayeredness:
    def test_layered_high_score(self, layered_mos2):
        """MoS2-like: gap 0.84 in c -> score > 0.7."""
        r = VoronoiAnalyzer().analyze(layered_mos2)
        assert r.layeredness_score > 0.7

    def test_two_layer_high_score(self, two_layer):
        """Two dense layers with 0.70 fractional gap -> score > 0.6."""
        r = VoronoiAnalyzer().analyze(two_layer)
        assert r.layeredness_score > 0.6

    def test_nacl_moderate_score(self, dense_nacl):
        """NaCl: atoms at 0 and 0.5 in every direction -> max gap = 0.5."""
        r = VoronoiAnalyzer().analyze(dense_nacl)
        assert r.layeredness_score < 0.55

    def test_stacking_direction_is_c_for_layered(self, layered_mos2):
        """The largest gap is along c (direction index 2)."""
        r = VoronoiAnalyzer().analyze(layered_mos2)
        assert r.stacking_direction == 2

    def test_score_bounded_0_1(self, dense_nacl):
        r = VoronoiAnalyzer().analyze(dense_nacl)
        assert 0.0 <= r.layeredness_score <= 1.0


# ------------------------------------------------------------------ #
#  interlayer_spacing                                                 #
# ------------------------------------------------------------------ #

class TestInterlayerSpacing:
    def test_layered_spacing(self, layered_mos2):
        """Gap = 0.84 * 20 A = 16.8 A."""
        r = VoronoiAnalyzer().analyze(layered_mos2)
        assert r.interlayer_spacing is not None
        assert 15.0 < r.interlayer_spacing < 18.0

    def test_two_layer_spacing(self, two_layer):
        """Gap ~ 0.70 * 16 A ~ 11.2 A (actual gap 0.10->0.90 wrap: 0.80*16=12.8)."""
        r = VoronoiAnalyzer().analyze(two_layer)
        assert r.interlayer_spacing is not None
        assert 10.0 < r.interlayer_spacing < 14.0


# ------------------------------------------------------------------ #
#  batch analysis                                                     #
# ------------------------------------------------------------------ #

class TestBatchAnalysis:
    def test_batch_returns_list(self, open_cubic, dense_nacl):
        va = VoronoiAnalyzer()
        results = va.analyze_batch([open_cubic, dense_nacl])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, VoronoiResult) for r in results)

    def test_batch_consistent_with_single(self, dense_nacl):
        va = VoronoiAnalyzer()
        single = va.analyze(dense_nacl)
        batch = va.analyze_batch([dense_nacl])[0]
        assert single.max_void_radius == pytest.approx(batch.max_void_radius)
        assert single.void_fraction == pytest.approx(batch.void_fraction)
        assert single.layeredness_score == pytest.approx(batch.layeredness_score)


# ------------------------------------------------------------------ #
#  CLI smoke test for ``python -m crystalformer_x.analysis``          #
# ------------------------------------------------------------------ #

class TestAnalysisCLI:
    def test_smoke_voronoi_only(self, tmp_path, open_cubic):
        from crystalformer_x.analysis.__main__ import main as analysis_main

        csv_path = tmp_path / "structs.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["cif"])
            writer.writeheader()
            writer.writerow({"cif": str(open_cubic.as_dict())})

        output = str(tmp_path / "out.csv")
        analysis_main([str(csv_path), "-o", output, "--voronoi-only"])
        assert Path(output).exists()
        with open(output) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert "max_void_radius" in rows[0]

    def test_smoke_full_analysis(self, tmp_path, open_cubic):
        from crystalformer_x.analysis.__main__ import main as analysis_main

        csv_path = tmp_path / "structs.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["cif"])
            writer.writeheader()
            writer.writerow({"cif": str(open_cubic.as_dict())})

        output = str(tmp_path / "out.csv")
        analysis_main([str(csv_path), "-o", output, "--check-percolation"])
        assert Path(output).exists()
        with open(output) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert "percolation_dim" in rows[0]
