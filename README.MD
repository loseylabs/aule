# Aule

A meta-evolutionary genetic algorithm framework where everything is an Individual and capable of evolution.
Based on my previous project, Finch.

## Installation

```bash
git clone https://github.com/yourusername/aule.git
cd aule
pip install -e .
```

## Quick Start

Evolve a string to match "hello world":

```python
from aule import Environment, EvaluationLayer, PopulateLayer, SortLayer, CapPopulationLayer
from aule.species import ListGenePool, ListPointCrossoverLayer, ListRandomMutationLayer

TARGET = list("hello world")
VOCAB = list("abcdefghijklmnopqrstuvwxyz ")

genepool = ListGenePool(vocab=VOCAB, length=len(TARGET))
fitness = lambda ind: sum(1 for a, b in zip(ind.genes, TARGET) if a == b)

env = Environment(layers=[
    PopulateLayer(genepool, size=500),
    ListPointCrossoverLayer(points=2, offspring_count=250),
    ListRandomMutationLayer(VOCAB, rate=0.02),
    EvaluationLayer(fitness),
    SortLayer(),
    CapPopulationLayer(500),
])

env.evolve(epochs=100)
print(''.join(env.best_ever.genes))  # hello world
```

## Philosophy

All components—environments, layers, operators—are Individuals that can evolve. This enables co-evolution at every level: mutation strategies compete within mutation layers, layers compete within environments, and environments compete within meta-environments.

## Species

Built-in representations for different problem types:

**Tensors** (`aule.species.tensor_evo`) - PyTorch tensors for numerical optimization
- `TensorIndividual`, `TensorGenePool`
- `PointCrossoverLayer`, `UniformCrossoverLayer`, `BlendCrossoverLayer`
- `GaussianMutationLayer`, `RandomMutationLayer`, `SwapMutationLayer`

**Lists** (`aule.species.list_evo`) - Sequences from any vocabulary
- `ListIndividual`, `ListGenePool`
- `ListPointCrossoverLayer`, `ListUniformCrossoverLayer`
- `ListRandomMutationLayer`, `ListSwapMutationLayer`, `ListShuffleMutationLayer`

## Layers

Evolution is defined as a pipeline of layers:

| Layer | Purpose |
|-------|---------|
| `PopulateLayer` | Generate initial population |
| `EvaluationLayer` | Compute fitness |
| `SelectionLayer` | Select parents for reproduction |
| `SortLayer` | Sort population by fitness |
| `CapPopulationLayer` | Limit population size |

## Selection Strategies

- `TournamentSelection` - Pick best from random subsets
- `RouletteSelection` - Probability proportional to fitness
- `RankSelection` - Probability based on rank
- `TruncationSelection` - Keep top percentage
- `ElitistSelection` - Preserve top n, tournament for rest

## Examples

Run the examples:

```bash
python examples/hello_world.py      # String evolution
python examples/fitness_comparison.py  # Compare fitness functions
```
