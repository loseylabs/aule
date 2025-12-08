"""Compare different fitness functions on tensor evolution."""
import matplotlib.pyplot as plt
from aule import Environment, EvaluationLayer, PopulateLayer, SortLayer, CapPopulationLayer
from aule.species import TensorGenePool, PointCrossoverLayer, GaussianMutationLayer

EPOCHS = 500

fitness_fns = {
    'mean': lambda ind: ind.genes.mean().item(),
    'min': lambda ind: ind.genes.min().item(),
    'max': lambda ind: ind.genes.max().item(),
}

def make_env(fitness_fn):
    genepool = TensorGenePool(length=20, low=0.0, high=1.0)
    return Environment(layers=[
        PopulateLayer(genepool, size=100),
        PointCrossoverLayer(points=2, offspring_count=50),
        GaussianMutationLayer(rate=0.1, std=0.1),
        EvaluationLayer(fitness_fn),
        SortLayer(),
        CapPopulationLayer(100),
    ])

results = {}
for name, fn in fitness_fns.items():
    print(f"\n=== Optimizing {name} ===")
    env = make_env(fn)
    env.evolve(EPOCHS)
    mean_over_time = [env.population[0].genes.mean().item() for _ in range(EPOCHS)]
    
    # Re-run to capture mean at each epoch
    env = make_env(fn)
    mean_history = []
    for _ in range(EPOCHS):
        for layer in env.layers:
            env.population = layer.execute(env.population)
        env.population.sort(key=lambda x: x.fitness, reverse=True)
        mean_history.append(env.population[0].genes.mean().item())
    
    results[name] = mean_history
    print(f"Final mean: {mean_history[-1]:.4f}")

plt.figure(figsize=(10, 6))
for name, history in results.items():
    plt.plot(history, label=f'optimize {name}')

plt.xlabel('Epoch')
plt.ylabel('Mean of Best Individual')
plt.title('Effect of Fitness Function on Gene Mean')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
