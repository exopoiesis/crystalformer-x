"""Built-in structure filters for the screening pipeline.

Each filter is a callable that accepts a ``pymatgen.Structure`` and
returns ``True`` if the structure passes.
"""

from abc import ABC, abstractmethod

import numpy as np

from crystalformer_x.analysis.percolation import PercolationAnalyzer
from crystalformer_x.analysis.voronoi import VoronoiAnalyzer


class Filter(ABC):
    """Base class for structure filters.

    Subclass and implement ``__call__`` to create custom filters.
    Set ``name`` to a short identifier used in summaries and specs.
    """

    name: str = "filter"

    @abstractmethod
    def __call__(self, structure) -> bool:
        """Return True if *structure* passes this filter."""


class ValidityFilter(Filter):
    """Reject structures with overlapping atoms.

    Parameters:
        min_dist: Minimum allowed interatomic distance (angstrom).
    """

    name = "validity"

    def __init__(self, min_dist: float = 0.5):
        self.min_dist = min_dist

    def __call__(self, structure) -> bool:
        if len(structure) < 2:
            return True
        dmat = structure.distance_matrix
        np.fill_diagonal(dmat, np.inf)
        return float(np.min(dmat)) >= self.min_dist


class VoronoiFilter(Filter):
    """Keep structures whose largest void exceeds a threshold.

    Parameters:
        r_min: Minimum max-void-radius (angstrom, default 1.0).
        r_probe: Probe radius for void-fraction computation (angstrom,
            default 0.4).  Independent of *r_min*.
        grid_resolution: Grid spacing for the Voronoi analysis.
    """

    name = "voronoi"

    def __init__(
        self,
        r_min: float = 1.0,
        r_probe: float = 0.4,
        grid_resolution: float = 0.25,
    ):
        self.r_min = r_min
        self._analyzer = VoronoiAnalyzer(
            r_probe=r_probe, grid_resolution=grid_resolution
        )

    def __call__(self, structure) -> bool:
        result = self._analyzer.analyze(structure)
        return result.max_void_radius >= self.r_min


class PercolationFilter(Filter):
    """Keep structures with enough percolating directions.

    Parameters:
        min_dim: Minimum percolation dimensionality (default 1).
        r_probe: Probe radius in angstrom (default 0.4).
        grid_resolution: Grid spacing for percolation analysis.
    """

    name = "percolation"

    def __init__(
        self,
        min_dim: int = 1,
        r_probe: float = 0.4,
        grid_resolution: float = 0.3,
    ):
        self.min_dim = min_dim
        self._analyzer = PercolationAnalyzer(
            r_probe=r_probe, grid_resolution=grid_resolution
        )

    def __call__(self, structure) -> bool:
        result = self._analyzer.analyze(structure)
        return result.percolation_dimensionality >= self.min_dim


class CompositionFilter(Filter):
    """Keep structures containing all specified elements.

    Parameters:
        elements: List of element symbols that must be present.
    """

    name = "composition"

    def __init__(self, elements: list[str] | None = None):
        self.elements = set(elements) if elements else set()

    def __call__(self, structure) -> bool:
        if not self.elements:
            return True
        present = {sp.symbol for sp in structure.species}
        return self.elements.issubset(present)


class DensityFilter(Filter):
    """Keep structures within a density range.

    Parameters:
        min_density: Lower bound (g/cm3), or None for no lower bound.
        max_density: Upper bound (g/cm3), or None for no upper bound.
    """

    name = "density"

    def __init__(
        self,
        min_density: float | None = None,
        max_density: float | None = None,
    ):
        self.min_density = min_density
        self.max_density = max_density

    def __call__(self, structure) -> bool:
        d = structure.density
        if self.min_density is not None and d < self.min_density:
            return False
        if self.max_density is not None and d > self.max_density:
            return False
        return True


# ------------------------------------------------------------------ #
#  Registry for string-spec resolution                                #
# ------------------------------------------------------------------ #

_FILTER_REGISTRY: dict[str, type[Filter]] = {
    "validity": ValidityFilter,
    "voronoi": VoronoiFilter,
    "percolation": PercolationFilter,
    "composition": CompositionFilter,
    "density": DensityFilter,
}


def resolve_filter(spec: str | Filter) -> Filter:
    """Resolve a filter from a string spec or pass through a Filter instance.

    String specs: ``"name"`` or ``"name:key=val,key=val"``.
    For CompositionFilter, elements are separated by ``+``:
    ``"composition:elements=Fe+S+Ni"``.
    """
    if isinstance(spec, Filter):
        return spec

    if ":" in spec:
        name, params_str = spec.split(":", 1)
    else:
        name, params_str = spec, ""

    name = name.strip()
    if name not in _FILTER_REGISTRY:
        raise ValueError(
            f"Unknown filter: {name!r}. "
            f"Available: {', '.join(_FILTER_REGISTRY)}"
        )

    cls = _FILTER_REGISTRY[name]

    if not params_str:
        return cls()

    kwargs = {}
    for item in params_str.split(","):
        key, _, val = item.strip().partition("=")
        key = key.strip()
        val = val.strip()
        # Type coercion
        if key == "elements":
            kwargs[key] = val.split("+")
        else:
            try:
                kwargs[key] = int(val)
            except ValueError:
                try:
                    kwargs[key] = float(val)
                except ValueError:
                    kwargs[key] = val

    return cls(**kwargs)
