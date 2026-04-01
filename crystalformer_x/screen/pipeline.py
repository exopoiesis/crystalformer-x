"""Screening pipeline -- chain filters and track funnel statistics."""

from dataclasses import dataclass, field

from crystalformer_x.screen.filters import Filter, resolve_filter


@dataclass
class ScreeningResult:
    """Result of running a screening pipeline.

    Attributes:
        input_count: Number of structures fed into the pipeline.
        survivors: Structures that passed all filters.
        survivor_indices: Indices into the original input list.
        stage_counts: Ordered dict mapping stage name to the count of
            structures remaining after that stage.
    """

    input_count: int
    survivors: list
    survivor_indices: list[int]
    stage_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable funnel summary."""
        lines = []
        prev = self.input_count
        for stage, count in self.stage_counts.items():
            drop = prev - count
            pct = 100 * count / self.input_count if self.input_count else 0
            marker = f" (-{drop})" if drop > 0 else ""
            lines.append(f"  {stage:<20s} {count:>6d}  ({pct:5.1f}%){marker}")
            prev = count
        header = f"Screening: {self.input_count} input -> {len(self.survivors)} passed"
        return header + "\n" + "\n".join(lines)


class ScreeningPipeline:
    """Chain multiple structure filters into a screening funnel.

    Parameters:
        filters: List of :class:`Filter` instances or string specs
            (e.g. ``"validity"``, ``"voronoi:r_min=1.0"``).

    Example::

        pipe = ScreeningPipeline([
            "validity",
            "voronoi:r_min=0.4",
            "percolation:min_dim=2,r_probe=0.4",
            CompositionFilter(elements=["Fe", "S"]),
        ])
        result = pipe.run(structures)
        print(result.summary())
    """

    def __init__(self, filters: list):
        self.filters: list[Filter] = [resolve_filter(f) for f in filters]

    def run(self, structures: list) -> ScreeningResult:
        """Run all filters sequentially.

        Returns a :class:`ScreeningResult` with surviving structures,
        their indices, and per-stage counts.
        """
        current = list(enumerate(structures))  # (index, structure)
        stage_counts: dict[str, int] = {"input": len(current)}

        for filt in self.filters:
            current = [
                (idx, s) for idx, s in current if filt(s)
            ]
            # Avoid overwriting counts when two filters share a name
            key = filt.name
            if key in stage_counts:
                n = sum(1 for k in stage_counts if k.startswith(key))
                key = f"{key}_{n}"
            stage_counts[key] = len(current)

        indices = [idx for idx, _ in current]
        survivors = [s for _, s in current]

        return ScreeningResult(
            input_count=len(structures),
            survivors=survivors,
            survivor_indices=indices,
            stage_counts=stage_counts,
        )
