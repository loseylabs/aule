"""Aule: A meta-evolutionary genetic algorithm framework.

All components are Individuals that can evolve, enabling co-evolution
at every level: operators compete within layers, layers compete within
environments, environments compete within meta-environments.
"""
import copy
from typing import List, Optional, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm


class Individual:
    """Base class for all evolvable components."""
    
    def __init__(self, fitness: float = 0.0):
        self.fitness = fitness
        self.age = 0
        self.evaluated = False


class GenePool(Individual):
    """Generates new individuals for a population."""
    
    def generate_individuals(self, amount: int) -> List[Individual]:
        raise NotImplementedError


class Selection(Individual):
    """Selects individuals from a population."""
    
    def select(self, individuals: List[Individual]) -> List[Individual]:
        raise NotImplementedError


class Layer(Individual):
    """Processes a population and returns the modified population."""
    
    def __init__(self, genepool: Optional[GenePool] = None):
        super().__init__()
        self.genepool = genepool
    
    def execute(self, individuals: List[Individual]) -> List[Individual]:
        raise NotImplementedError


class CompetitionLayer(Layer):
    """Assigns fitness through head-to-head competition."""
    
    def __init__(self, selector: Selection, compete_fn: Callable):
        super().__init__()
        self.selector = selector
        self.compete = compete_fn


class Environment(Individual):
    """Orchestrates evolution through a pipeline of layers."""
    
    def __init__(self, layers: Optional[List[Layer]] = None):
        super().__init__()
        self.layers = layers or []
        self.population: List[Individual] = []
        self.best_ever: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.population_history: List[int] = []
        self.avg_age_history: List[float] = []
    
    def evolve(self, epochs: int):
        """Run evolution for the given number of epochs."""
        pbar = tqdm(range(epochs), desc="Evolving", unit="epoch")
        for _ in pbar:
            for layer in self.layers:
                self.population = layer.execute(self.population)
            
            for ind in self.population:
                ind.age += 1
            
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best = self.population[0]
            avg_age = sum(ind.age for ind in self.population) / len(self.population)
            
            self.fitness_history.append(best.fitness)
            self.population_history.append(len(self.population))
            self.avg_age_history.append(avg_age)
            
            if not self.best_ever or best.fitness > self.best_ever.fitness:
                self.best_ever = copy.deepcopy(best)
            
            pbar.set_postfix(best=f"{self.best_ever.fitness:.4f}", 
                           pop=len(self.population), 
                           avg_age=f"{avg_age:.1f}")
    
    def plot_dynamics(self):
        """Plot evolution dynamics."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        axes[0].plot(self.fitness_history)
        axes[0].set_ylabel('Best Fitness')
        
        axes[1].plot(self.population_history)
        axes[1].set_ylabel('Population Size')
        
        axes[2].plot(self.avg_age_history)
        axes[2].set_ylabel('Average Age')
        axes[2].set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.show()
