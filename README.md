# CrystalFormer-X

Post-generation analysis, screening, and guided sampling toolkit for [CrystalFormer](https://github.com/deepmodeling/CrystalFormer).

CrystalFormer generates crystal structures. **CrystalFormer-X** helps you figure out which ones are useful.

```
CrystalFormer → structures.csv → CrystalFormer-X → candidates.csv
                 (thousands)       (analysis +        (tens)
                                    screening)
```

## Installation

```bash
pip install crystalformer-x
```

Dependencies: `numpy`, `scipy`, `pymatgen`. No JAX required.

## Full Pipeline Example

```bash
# 1. Generate structures with CrystalFormer (upstream)
python main.py --optimizer none --restore_path model.pkl \
    --spacegroup 194 --num_samples 5000 \
    --composition_bias "Fe:2.0,S:1.5"

# 2. Convert to CIF
python scripts/awl2struct.py --output_path . --label 194

# 3. Analyze: void characterization + channel detection
python -m crystalformer_x.analysis output_194_struct.csv \
    --r-probe 0.4 --check-percolation -o analyzed.csv

# 4. Screen for layered ionic conductors
python -m crystalformer_x.screen output_194_struct.csv \
    --filters validity,voronoi,percolation,composition \
    --r-probe 0.4 --percolation-dim 2 --elements Fe S \
    -o candidates.csv

# 5. Relax survivors only (saves compute!)
python scripts/mlff_relax.py --restore_path . \
    --filename candidates.csv --model orb-v3-conservative-inf-mpa ...
```

## Modules

### Structural Analysis

Voronoi-based characterization of crystal structures: void size, void fraction, layeredness, interlayer spacing, and percolation/channel analysis.

```python
from crystalformer_x.analysis import VoronoiAnalyzer, PercolationAnalyzer

va = VoronoiAnalyzer(r_probe=0.4)
vr = va.analyze(structure)  # pymatgen Structure
# vr.max_void_radius    → 1.82 A
# vr.void_fraction      → 0.34
# vr.layeredness_score  → 0.91
# vr.interlayer_spacing → 2.45 A

pa = PercolationAnalyzer(r_probe=0.4)
pr = pa.analyze(structure)
# pr.percolation_dimensionality → 2 (layered conductor)
# pr.percolates_a/b/c           → True/True/False
# pr.min_bottleneck              → 0.62 A
```

```bash
python -m crystalformer_x.analysis structures.csv --r-probe 0.4 --check-percolation
```

Full documentation: [analysis/README.md](crystalformer_x/analysis/README.md)

### Screening Pipeline

Pluggable filter chain for batch screening. Chain multiple criteria and get a funnel summary showing where candidates drop off.

```python
from crystalformer_x.screen import ScreeningPipeline, CompositionFilter

pipeline = ScreeningPipeline([
    "validity",
    "voronoi:r_min=1.0",
    "percolation:min_dim=2,r_probe=0.4",
    CompositionFilter(elements=["Fe", "S"]),
])

result = pipeline.run(structures)
print(result.summary())
# Screening: 5000 input -> 127 passed
#   input                  5000  (100.0%)
#   validity               4812  ( 96.2%) (-188)
#   voronoi                2034  ( 40.7%) (-2778)
#   percolation             389  (  7.8%) (-1645)
#   composition             127  (  2.5%) (-262)
```

**Built-in filters:** `validity`, `voronoi`, `percolation`, `composition`, `density`

**Custom filters:**

```python
from crystalformer_x.screen import Filter

class StabilityFilter(Filter):
    name = "stability"
    def __init__(self, model, threshold=0.05):
        self.model = model
        self.threshold = threshold
    def __call__(self, structure) -> bool:
        return self.model.predict(structure) < self.threshold
```

```bash
python -m crystalformer_x.screen structures.csv \
    --filters validity,voronoi,percolation,composition \
    --r-probe 0.4 --percolation-dim 2 --elements Fe S \
    -o candidates.csv
```

Full documentation: [screen/README.md](crystalformer_x/screen/README.md)

## Compositional Guided Sampling

The `--composition_bias` feature lives in the upstream CrystalFormer repo (see [PR #14](https://github.com/deepmodeling/CrystalFormer/pull/14)). It adds a soft logit bias during atom-type sampling, nudging the model toward desired elements without retraining.

```bash
python main.py --optimizer none --restore_path model.pkl \
    --composition_bias "Fe:3.0,S:3.0,Ni:2.0" \
    --num_samples 1000 --K 10
```

| Weight | Effect |
|--------|--------|
| 0.0 | No change (default) |
| 1.0-2.0 | Gentle nudge |
| 3.0-5.0 | Strong bias — most atoms will be target elements |
| -1.0 to -3.0 | Suppresses without fully blocking |

## How to cite

If you use CrystalFormer-X, please cite the upstream CrystalFormer paper:

```bibtex
@article{cao2024space,
      title={Space Group Informed Transformer for Crystalline Materials Generation},
      author={Zhendong Cao and Xiaoshan Luo and Jian Lv and Lei Wang},
      year={2024},
      eprint={2403.15734},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci}
}
```

## License

MIT
