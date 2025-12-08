"""Evolve a string to match 'hello world'."""
from aule import Environment, EvaluationLayer, PopulateLayer, SortLayer, CapPopulationLayer, SelectionLayer, TournamentSelection
from aule.species import ListGenePool, ListPointCrossoverLayer, ListRandomMutationLayer

TARGET = list("hello world")
VOCAB = list("abcdefghijklmnopqrstuvwxyz ")

genepool = ListGenePool(vocab=VOCAB, length=len(TARGET))

def fitness(ind):
    return sum(1 for a, b in zip(ind.genes, TARGET) if a == b)

env = Environment(layers=[
    PopulateLayer(genepool, size=500),
    EvaluationLayer(fitness),
    SelectionLayer(TournamentSelection(tournament_size=5)),
    ListPointCrossoverLayer(points=2, offspring_count=250),
    ListRandomMutationLayer(VOCAB, rate=0.02),
    EvaluationLayer(fitness),
    SortLayer(),
    CapPopulationLayer(500),
])

env.evolve(epochs=30)

result = ''.join(env.best_ever.genes)
print(f"\nTarget: {''.join(TARGET)}")
print(f"Result: {result}")
print(f"Fitness: {env.best_ever.fitness}/{len(TARGET)}")

env.plot_dynamics()