# Aule

A meta-evolutionary genetic algorithm framework where everything is an Individual.

## Installation

```bash
pip install -e .
```

## Usage

```python
from aule import Individual, Environment, TournamentSelection
from aule.species import TensorIndividual, TensorGenePool, PointCrossover, GaussianMutation

# Create a gene pool for float tensors
pool = TensorGenePool(length=10, low=0.0, high=1.0)

# Generate individuals
population = pool.generate_individuals(100)
```

## Philosophy

All components—environments, layers, operators—are Individuals that can evolve. This enables co-evolution at every level: mutation strategies compete within mutation layers, layers compete within environments, and environments compete within meta-environments.
