"""List evolution - evolve sequences from any vocabulary."""
import random
from typing import List, Any
from aule.abstract import Individual, GenePool, Layer


class ListIndividual(Individual):
    """An individual represented as a list of items from a vocabulary."""
    
    def __init__(self, genes: List[Any], fitness: float = 0.0):
        super().__init__(fitness)
        self.genes = genes


class ListGenePool(GenePool):
    """Generates random list individuals from a vocabulary."""
    
    def __init__(self, vocab: List[Any], length: int):
        super().__init__()
        self.vocab = vocab
        self.length = length
    
    def generate_individuals(self, amount: int) -> List[ListIndividual]:
        return [ListIndividual([random.choice(self.vocab) for _ in range(self.length)]) 
                for _ in range(amount)]


# ============================================================================
# Crossover Layers
# ============================================================================

class ListCrossoverLayer(Layer):
    """Base crossover layer for lists."""
    
    def __init__(self, offspring_count: int = 50):
        super().__init__()
        self.offspring_count = offspring_count
    
    def execute(self, individuals: List[ListIndividual]) -> List[ListIndividual]:
        offspring = [self._cross(*random.sample(individuals, 2)) for _ in range(self.offspring_count)]
        return individuals + offspring
    
    def _cross(self, a: ListIndividual, b: ListIndividual) -> ListIndividual:
        raise NotImplementedError


class ListPointCrossoverLayer(ListCrossoverLayer):
    """Single/multi-point crossover for lists."""
    
    def __init__(self, points: int = 1, offspring_count: int = 50):
        super().__init__(offspring_count)
        self.points = points
    
    def _cross(self, a: ListIndividual, b: ListIndividual) -> ListIndividual:
        length = len(a.genes)
        indices = sorted(random.sample(range(1, length), min(self.points, length - 1)))
        
        child = a.genes.copy()
        swap = False
        prev = 0
        for idx in indices:
            if swap:
                child[prev:idx] = b.genes[prev:idx]
            swap = not swap
            prev = idx
        if swap:
            child[prev:] = b.genes[prev:]
        
        return ListIndividual(child)


class ListUniformCrossoverLayer(ListCrossoverLayer):
    """Uniform crossover - each gene randomly from either parent."""
    
    def _cross(self, a: ListIndividual, b: ListIndividual) -> ListIndividual:
        child = [random.choice([a.genes[i], b.genes[i]]) for i in range(len(a.genes))]
        return ListIndividual(child)


# ============================================================================
# Mutation Layers
# ============================================================================

class ListRandomMutationLayer(Layer):
    """Replace random genes with new random values from vocabulary."""
    
    def __init__(self, vocab: List[Any], rate: float = 0.1):
        super().__init__()
        self.vocab = vocab
        self.rate = rate
    
    def execute(self, individuals: List[ListIndividual]) -> List[ListIndividual]:
        for ind in individuals:
            for i in range(len(ind.genes)):
                if random.random() < self.rate:
                    ind.genes[i] = random.choice(self.vocab)
        return individuals


class ListSwapMutationLayer(Layer):
    """Swap two random positions."""
    
    def execute(self, individuals: List[ListIndividual]) -> List[ListIndividual]:
        for ind in individuals:
            i, j = random.sample(range(len(ind.genes)), 2)
            ind.genes[i], ind.genes[j] = ind.genes[j], ind.genes[i]
        return individuals


class ListShuffleMutationLayer(Layer):
    """Shuffle a random segment of the list."""
    
    def __init__(self, max_segment: int = 5):
        super().__init__()
        self.max_segment = max_segment
    
    def execute(self, individuals: List[ListIndividual]) -> List[ListIndividual]:
        for ind in individuals:
            length = len(ind.genes)
            seg_len = min(random.randint(2, self.max_segment), length)
            start = random.randint(0, length - seg_len)
            segment = ind.genes[start:start + seg_len]
            random.shuffle(segment)
            ind.genes[start:start + seg_len] = segment
        return individuals

