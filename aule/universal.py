"""Universal GA components - work on any individual type regardless of representation."""
import torch
import random
from typing import List, Callable
from aule.abstract import Individual, Selection, Layer, GenePool, CompetitionLayer


def tournament_winner(individuals: List[Individual], size: int) -> Individual:
    return max(random.sample(individuals, size), key=lambda x: x.fitness)


class PopulateLayer(Layer):
    """Generate individuals from a genepool if population is empty."""
    
    def __init__(self, genepool: GenePool, size: int):
        super().__init__(genepool=genepool)
        self.size = size
    
    def execute(self, individuals: List[Individual]) -> List[Individual]:
        return individuals or self.genepool.generate_individuals(self.size)


# ============================================================================
# Selection Strategies
# ============================================================================

class TournamentSelection(Selection):
    """Tournament selection - pick best from random subsets."""
    
    def __init__(self, tournament_size: int = 3):
        super().__init__()
        self.tournament_size = tournament_size
    
    def select(self, individuals: List[Individual]) -> List[Individual]:
        return [tournament_winner(individuals, self.tournament_size) for _ in range(len(individuals))]


class RouletteSelection(Selection):
    """Roulette wheel selection - probability proportional to fitness."""
    
    def select(self, individuals: List[Individual]) -> List[Individual]:
        fitnesses = torch.tensor([ind.fitness for ind in individuals])
        fitnesses = fitnesses - fitnesses.min() + 1e-6
        probs = fitnesses / fitnesses.sum()
        indices = torch.multinomial(probs, len(individuals), replacement=True)
        return [individuals[i] for i in indices]


class RankSelection(Selection):
    """Rank-based selection - probability based on rank, not raw fitness."""
    
    def __init__(self, pressure: float = 1.5):
        super().__init__()
        self.pressure = pressure
    
    def select(self, individuals: List[Individual]) -> List[Individual]:
        n = len(individuals)
        sorted_indices = sorted(range(n), key=lambda i: individuals[i].fitness)
        ranks = torch.zeros(n)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank
        
        probs = (2 - self.pressure + 2 * (self.pressure - 1) * ranks / (n - 1)) / n
        probs = probs / probs.sum()
        indices = torch.multinomial(probs, n, replacement=True)
        return [individuals[i] for i in indices]


class TruncationSelection(Selection):
    """Truncation selection - keep only the top percentage."""
    
    def __init__(self, keep_ratio: float = 0.5):
        super().__init__()
        self.keep_ratio = keep_ratio
    
    def select(self, individuals: List[Individual]) -> List[Individual]:
        sorted_inds = sorted(individuals, key=lambda x: x.fitness, reverse=True)
        keep_count = int(len(individuals) * self.keep_ratio)
        return [sorted_inds[i % keep_count] for i in range(len(individuals))]


class ElitistSelection(Selection):
    """Elitist selection - guarantee top n survive, rest via tournament."""
    
    def __init__(self, elite_count: int = 1, tournament_size: int = 3):
        super().__init__()
        self.elite_count = elite_count
        self.tournament_size = tournament_size
    
    def select(self, individuals: List[Individual]) -> List[Individual]:
        sorted_inds = sorted(individuals, key=lambda x: x.fitness, reverse=True)
        elite = list(sorted_inds[:self.elite_count])
        remaining = len(individuals) - self.elite_count
        elite.extend(tournament_winner(individuals, self.tournament_size) for _ in range(remaining))
        return elite


class RandomSelection(Selection):
    """Random selection - uniform random sampling."""
    
    def select(self, individuals: List[Individual]) -> List[Individual]:
        return random.choices(individuals, k=len(individuals))


class PairSelection(Selection):
    """Select random pairs for competition or crossover."""
    
    def select(self, individuals: List[Individual]) -> List[Individual]:
        shuffled = individuals.copy()
        random.shuffle(shuffled)
        return shuffled


# ============================================================================
# Layers
# ============================================================================

class PairwiseCompetitionLayer(CompetitionLayer):
    """Competition where pairs face off and winners gain fitness."""
    
    def __init__(self, compete_fn: Callable[[Individual, Individual], Individual], rounds: int = 1):
        super().__init__(PairSelection(), compete_fn)
        self.rounds = rounds
    
    def execute(self, individuals: List[Individual]) -> List[Individual]:
        wins = {id(ind): 0 for ind in individuals}
        
        for _ in range(self.rounds):
            shuffled = individuals.copy()
            random.shuffle(shuffled)
            for i in range(0, len(shuffled), 2):
                a, b = shuffled[i], shuffled[i + 1]
                winner = self.compete(a, b)
                wins[id(winner)] += 1
        
        total = self.rounds * (len(individuals) // 2) * 2 // len(individuals)
        for ind in individuals:
            ind.fitness = wins[id(ind)] / total
        
        return individuals


class EvaluationLayer(Layer):
    """Evaluate fitness for unevaluated individuals."""
    
    def __init__(self, evaluate_fn: Callable[[Individual], float]):
        super().__init__()
        self.evaluate = evaluate_fn
    
    def execute(self, individuals: List[Individual]) -> List[Individual]:
        for ind in individuals:
            if not ind.evaluated:
                ind.fitness = self.evaluate(ind)
                ind.evaluated = True
        return individuals


class SelectionLayer(Layer):
    """Apply a selection strategy to the population."""
    
    def __init__(self, selector: Selection):
        super().__init__()
        self.selector = selector
    
    def execute(self, individuals: List[Individual]) -> List[Individual]:
        return self.selector.select(individuals)


class SortLayer(Layer):
    """Sort population by fitness (best first)."""
    
    def execute(self, individuals: List[Individual]) -> List[Individual]:
        return sorted(individuals, key=lambda x: x.fitness, reverse=True)


class CapPopulationLayer(Layer):
    """Cap population size by removing worst individuals (assumes sorted)."""
    
    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = max_size
    
    def execute(self, individuals: List[Individual]) -> List[Individual]:
        return individuals[:self.max_size]

