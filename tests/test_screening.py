"""Tests for crystalformer_x.screen -- ScreeningPipeline and built-in filters."""

import csv
from pathlib import Path

import pytest
import numpy as np

pytest.importorskip("pymatgen", reason="pymatgen required for screening tests")

from pymatgen.core import Structure, Lattice

from crystalformer_x.screen import ScreeningPipeline, ScreeningResult
from crystalformer_x.screen.filters import (
    Filter,
    ValidityFilter,
    VoronoiFilter,
    PercolationFilter,
    CompositionFilter,
    DensityFilter,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #

@pytest.fixture
def open_cubic():
    """Large voids, low density, single element Si."""
    return Structure(Lattice.cubic(8.0), ["Si"], [[0.0, 0.0, 0.0]])


@pytest.fixture
def layered_mos2():
    """Layered, percolating channels, Mo+S composition."""
    return Structure(
        Lattice.from_parameters(3.2, 3.2, 20.0, 90, 90, 120),
        ["Mo", "S", "S"],
        [[0.0, 0.0, 0.0], [0.333, 0.667, 0.08], [0.333, 0.667, 0.92]],
    )


@pytest.fixture
def dense_nacl():
    """Dense NaCl, small voids."""
    return Structure.from_spacegroup(
        225, Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0, 0]]
    )


@pytest.fixture
def overlapping():
    """Two atoms at the same position -- invalid structure."""
    return Structure(
        Lattice.cubic(5.0),
        ["Fe", "Fe"],
        [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]],  # ~0.005 A apart
    )


@pytest.fixture
def all_structures(open_cubic, layered_mos2, dense_nacl):
    return [open_cubic, layered_mos2, dense_nacl]


# ------------------------------------------------------------------ #
#  Filter base class                                                  #
# ------------------------------------------------------------------ #

class TestFilterBase:
    def test_custom_filter_works(self, open_cubic):
        """User-defined filter subclass integrates with pipeline."""

        class AlwaysPass(Filter):
            name = "always_pass"

            def __call__(self, structure) -> bool:
                return True

        f = AlwaysPass()
        assert f(open_cubic) is True

    def test_custom_filter_rejects(self, open_cubic):
        class AlwaysFail(Filter):
            name = "always_fail"

            def __call__(self, structure) -> bool:
                return False

        f = AlwaysFail()
        assert f(open_cubic) is False


# ------------------------------------------------------------------ #
#  ValidityFilter                                                     #
# ------------------------------------------------------------------ #

class TestValidityFilter:
    def test_normal_structure_passes(self, dense_nacl):
        f = ValidityFilter(min_dist=0.5)
        assert f(dense_nacl) is True

    def test_overlapping_fails(self, overlapping):
        f = ValidityFilter(min_dist=0.5)
        assert f(overlapping) is False

    def test_custom_threshold(self, dense_nacl):
        """NaCl nearest-neighbor ~ 2.82 A. Threshold 3.0 should fail."""
        f = ValidityFilter(min_dist=3.0)
        assert f(dense_nacl) is False

    def test_single_atom_passes(self, open_cubic):
        """Single atom -- no pairs to check, should pass."""
        f = ValidityFilter()
        assert f(open_cubic) is True


# ------------------------------------------------------------------ #
#  VoronoiFilter                                                      #
# ------------------------------------------------------------------ #

class TestVoronoiFilter:
    def test_open_passes(self, open_cubic):
        f = VoronoiFilter(r_min=1.0)
        assert f(open_cubic) is True

    def test_dense_fails_high_threshold(self, dense_nacl):
        f = VoronoiFilter(r_min=3.0)
        assert f(dense_nacl) is False

    def test_default_threshold(self, open_cubic):
        """Default r_min=1.0 A."""
        f = VoronoiFilter()
        assert f(open_cubic) is True


# ------------------------------------------------------------------ #
#  PercolationFilter                                                  #
# ------------------------------------------------------------------ #

class TestPercolationFilter:
    def test_open_percolates(self, open_cubic):
        f = PercolationFilter(min_dim=1, r_probe=1.0)
        assert f(open_cubic) is True

    def test_dense_blocked(self, dense_nacl):
        f = PercolationFilter(min_dim=1, r_probe=2.0)
        assert f(dense_nacl) is False

    def test_min_dim_3(self, open_cubic):
        f = PercolationFilter(min_dim=3, r_probe=1.0)
        assert f(open_cubic) is True


# ------------------------------------------------------------------ #
#  CompositionFilter                                                  #
# ------------------------------------------------------------------ #

class TestCompositionFilter:
    def test_contains_required_elements(self, layered_mos2):
        f = CompositionFilter(elements=["Mo", "S"])
        assert f(layered_mos2) is True

    def test_missing_element_fails(self, layered_mos2):
        f = CompositionFilter(elements=["Fe"])
        assert f(layered_mos2) is False

    def test_partial_match_fails(self, layered_mos2):
        """Must contain ALL specified elements."""
        f = CompositionFilter(elements=["Mo", "Fe"])
        assert f(layered_mos2) is False

    def test_empty_elements_passes_all(self, dense_nacl):
        f = CompositionFilter(elements=[])
        assert f(dense_nacl) is True


# ------------------------------------------------------------------ #
#  DensityFilter                                                      #
# ------------------------------------------------------------------ #

class TestDensityFilter:
    def test_within_range_passes(self, dense_nacl):
        """NaCl density ~ 2.16 g/cm3."""
        f = DensityFilter(min_density=1.0, max_density=3.0)
        assert f(dense_nacl) is True

    def test_below_range_fails(self, open_cubic):
        """Si in 8 A cell -- very low density."""
        f = DensityFilter(min_density=2.0)
        assert f(open_cubic) is False

    def test_above_range_fails(self, dense_nacl):
        f = DensityFilter(max_density=0.5)
        assert f(dense_nacl) is False

    def test_no_bounds_passes_all(self, open_cubic):
        f = DensityFilter()
        assert f(open_cubic) is True


# ------------------------------------------------------------------ #
#  ScreeningPipeline -- core                                          #
# ------------------------------------------------------------------ #

class TestPipelineCore:
    def test_empty_pipeline_passes_all(self, all_structures):
        pipe = ScreeningPipeline([])
        result = pipe.run(all_structures)
        assert len(result.survivors) == 3

    def test_single_filter(self, all_structures):
        pipe = ScreeningPipeline([CompositionFilter(elements=["Mo"])])
        result = pipe.run(all_structures)
        # Only layered_mos2 has Mo
        assert len(result.survivors) == 1

    def test_chained_filters_narrow(self, all_structures):
        pipe = ScreeningPipeline([
            ValidityFilter(),
            VoronoiFilter(r_min=3.0),  # dense NaCl fails here
        ])
        result = pipe.run(all_structures)
        # open_cubic and layered_mos2 pass; dense_nacl fails voronoi
        assert len(result.survivors) == 2

    def test_all_filtered_out(self, all_structures):
        pipe = ScreeningPipeline([CompositionFilter(elements=["Zr"])])
        result = pipe.run(all_structures)
        assert len(result.survivors) == 0

    def test_empty_input(self):
        pipe = ScreeningPipeline([ValidityFilter()])
        result = pipe.run([])
        assert len(result.survivors) == 0
        assert result.input_count == 0


# ------------------------------------------------------------------ #
#  ScreeningResult                                                    #
# ------------------------------------------------------------------ #

class TestScreeningResult:
    def test_result_has_fields(self, all_structures):
        pipe = ScreeningPipeline([ValidityFilter()])
        result = pipe.run(all_structures)
        assert isinstance(result, ScreeningResult)
        assert hasattr(result, "input_count")
        assert hasattr(result, "survivors")
        assert hasattr(result, "survivor_indices")
        assert hasattr(result, "stage_counts")

    def test_input_count(self, all_structures):
        result = ScreeningPipeline([]).run(all_structures)
        assert result.input_count == 3

    def test_survivor_indices_correct(self, all_structures):
        """Indices should point back to original input list."""
        pipe = ScreeningPipeline([CompositionFilter(elements=["Mo"])])
        result = pipe.run(all_structures)
        # all_structures = [open_cubic, layered_mos2, dense_nacl]
        # Only index 1 (layered_mos2) has Mo
        assert result.survivor_indices == [1]

    def test_stage_counts_tracks_funnel(self, all_structures):
        pipe = ScreeningPipeline([
            ValidityFilter(),
            CompositionFilter(elements=["Na", "Cl"]),
        ])
        result = pipe.run(all_structures)
        assert result.stage_counts["input"] == 3
        assert result.stage_counts["validity"] == 3  # all valid
        assert result.stage_counts["composition"] == 1  # only NaCl

    def test_summary_is_string(self, all_structures):
        result = ScreeningPipeline([ValidityFilter()]).run(all_structures)
        s = result.summary()
        assert isinstance(s, str)
        assert "3" in s  # input count should appear


# ------------------------------------------------------------------ #
#  String spec parsing                                                #
# ------------------------------------------------------------------ #

class TestCompositionFilterOxidationStates:
    """Regression: CompositionFilter must work with Species/oxidation states."""

    def test_oxidized_species_match(self):
        struct = Structure(
            Lattice.cubic(5.0),
            ["Fe2+", "O2-"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        f = CompositionFilter(elements=["Fe", "O"])
        assert f(struct) is True

    def test_mixed_species_and_elements(self):
        struct = Structure(
            Lattice.cubic(5.0),
            ["Fe2+", "S"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        f = CompositionFilter(elements=["Fe"])
        assert f(struct) is True


class TestStringSpecParsing:
    def test_validity_string(self, dense_nacl):
        pipe = ScreeningPipeline(["validity"])
        result = pipe.run([dense_nacl])
        assert len(result.survivors) == 1

    def test_voronoi_with_params(self, open_cubic):
        pipe = ScreeningPipeline(["voronoi:r_min=1.0"])
        result = pipe.run([open_cubic])
        assert len(result.survivors) == 1

    def test_percolation_with_params(self, open_cubic):
        pipe = ScreeningPipeline(["percolation:min_dim=1,r_probe=1.0"])
        result = pipe.run([open_cubic])
        assert len(result.survivors) == 1

    def test_composition_with_elements(self, layered_mos2):
        pipe = ScreeningPipeline(["composition:elements=Mo+S"])
        result = pipe.run([layered_mos2])
        assert len(result.survivors) == 1

    def test_density_with_range(self, dense_nacl):
        pipe = ScreeningPipeline(["density:min_density=1.0,max_density=3.0"])
        result = pipe.run([dense_nacl])
        assert len(result.survivors) == 1

    def test_mixed_strings_and_objects(self, all_structures):
        """Can mix string specs and Filter instances."""
        pipe = ScreeningPipeline([
            "validity",
            VoronoiFilter(r_min=1.0),
        ])
        result = pipe.run(all_structures)
        assert isinstance(result, ScreeningResult)

    def test_unknown_filter_raises(self):
        with pytest.raises(ValueError, match="Unknown filter"):
            ScreeningPipeline(["nonexistent_filter"])

    def test_scientific_notation_float(self, open_cubic):
        """Regression: '1e-3' must parse as float, not stay as string."""
        pipe = ScreeningPipeline(["voronoi:r_min=1e-3"])
        result = pipe.run([open_cubic])
        assert len(result.survivors) == 1

    def test_negative_scientific_notation(self, dense_nacl):
        pipe = ScreeningPipeline(["density:min_density=-1e2"])
        result = pipe.run([dense_nacl])
        assert len(result.survivors) == 1

    def test_bad_numeric_spec_raises_on_use(self, open_cubic):
        """Non-numeric value for a numeric param causes error at runtime."""
        pipe = ScreeningPipeline(["voronoi:r_min=abc"])
        with pytest.raises(TypeError):
            pipe.run([open_cubic])


# ------------------------------------------------------------------ #
#  CLI smoke tests                                                    #
# ------------------------------------------------------------------ #

def _make_csv(tmp_path, rows=None, fieldnames=None):
    """Helper: write a CSV with a 'cif' column from Structure objects."""
    if rows is None:
        s = Structure(Lattice.cubic(5.0), ["Si"], [[0, 0, 0]])
        rows = [{"cif": str(s.as_dict())}]
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    csv_path = tmp_path / "test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(csv_path)


class TestScreenCLI:
    """CLI-level tests for ``python -m crystalformer_x.screen``."""

    def test_smoke_validity(self, tmp_path):
        from crystalformer_x.screen.__main__ import main
        csv_path = _make_csv(tmp_path)
        output = str(tmp_path / "out.csv")
        main([csv_path, "-o", output, "--filters", "validity"])
        assert Path(output).exists()
        with open(output) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1

    def test_composition_without_elements_exits(self, tmp_path):
        from crystalformer_x.screen.__main__ import main
        csv_path = _make_csv(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            main([csv_path, "--filters", "composition"])
        assert exc_info.value.code == 2

    def test_unknown_filter_exits(self, tmp_path):
        from crystalformer_x.screen.__main__ import main
        csv_path = _make_csv(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            main([csv_path, "--filters", "bogus"])
        assert exc_info.value.code == 2

    def test_missing_cif_column_exits(self, tmp_path):
        from crystalformer_x.screen.__main__ import main
        csv_path = _make_csv(
            tmp_path,
            rows=[{"other": "data"}],
            fieldnames=["other"],
        )
        with pytest.raises(SystemExit) as exc_info:
            main([csv_path, "--filters", "validity"])
        assert exc_info.value.code == 1

    def test_malformed_cif_row_skipped(self, tmp_path):
        from crystalformer_x.screen.__main__ import main
        s = Structure(Lattice.cubic(5.0), ["Si"], [[0, 0, 0]])
        csv_path = _make_csv(tmp_path, rows=[
            {"cif": str(s.as_dict())},
            {"cif": "NOT_VALID_DICT"},
        ])
        output = str(tmp_path / "out.csv")
        main([csv_path, "-o", output, "--filters", "validity"])
        assert Path(output).exists()
        with open(output) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
