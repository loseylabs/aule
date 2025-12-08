"""Tensor evolution using PyTorch - supports int and float dtypes."""
import torch
import random
from typing import List
from aule.abstract import Individual, GenePool, Layer

INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.long, torch.short)


def random_tensor(shape, low: float, high: float, dtype: torch.dtype, device: str) -> torch.Tensor:
    if dtype in INT_DTYPES:
        return torch.randint(int(low), int(high), shape, dtype=dtype, device=device)
    return torch.rand(shape, dtype=dtype, device=device) * (high - low) + low


class TensorIndividual(Individual):
    """An individual represented as a 1D tensor."""
    
    def __init__(self, genes: torch.Tensor, fitness: float = 0.0):
        super().__init__(fitness)
        self.genes = genes


class TensorGenePool(GenePool):
    """Generates random tensor individuals."""
    
    def __init__(self, length: int, low: float, high: float, 
                 dtype: torch.dtype = torch.float32, device: str = 'cpu'):
        super().__init__()
        self.length = length
        self.low = low
        self.high = high
        self.dtype = dtype
        self.device = device
    
    def generate_individuals(self, amount: int) -> List[TensorIndividual]:
        return [TensorIndividual(random_tensor((self.length,), self.low, self.high, self.dtype, self.device)) 
                for _ in range(amount)]


# ============================================================================
# Crossover Layers
# ============================================================================

class CrossoverLayer(Layer):
    """Base crossover layer - generates offspring and adds to population."""
    
    def __init__(self, offspring_count: int = 50):
        super().__init__()
        self.offspring_count = offspring_count
    
    def execute(self, individuals: List[TensorIndividual]) -> List[TensorIndividual]:
        offspring = [self._cross(*random.sample(individuals, 2)) for _ in range(self.offspring_count)]
        return individuals + offspring
    
    def _cross(self, a: TensorIndividual, b: TensorIndividual) -> TensorIndividual:
        raise NotImplementedError


class PointCrossoverLayer(CrossoverLayer):
    """Single/multi-point crossover."""
    
    def __init__(self, points: int = 1, offspring_count: int = 50):
        super().__init__(offspring_count)
        self.points = points
    
    def _cross(self, a: TensorIndividual, b: TensorIndividual) -> TensorIndividual:
        length = a.genes.shape[0]
        indices = torch.sort(torch.randperm(length - 1)[:self.points] + 1).values
        
        child_genes = a.genes.clone()
        swap = False
        prev = 0
        for idx in indices:
            if swap:
                child_genes[prev:idx] = b.genes[prev:idx]
            swap = not swap
            prev = idx
        if swap:
            child_genes[prev:] = b.genes[prev:]
        
        return TensorIndividual(child_genes)


class UniformCrossoverLayer(CrossoverLayer):
    """Uniform crossover - each gene randomly from either parent."""
    
    def _cross(self, a: TensorIndividual, b: TensorIndividual) -> TensorIndividual:
        mask = torch.randint(0, 2, a.genes.shape, dtype=torch.bool, device=a.genes.device)
        return TensorIndividual(torch.where(mask, a.genes, b.genes))


class BlendCrossoverLayer(CrossoverLayer):
    """Blend crossover - interpolate between parents."""
    
    def __init__(self, alpha: float = 0.5, offspring_count: int = 50):
        super().__init__(offspring_count)
        self.alpha = alpha
    
    def _cross(self, a: TensorIndividual, b: TensorIndividual) -> TensorIndividual:
        t = torch.rand(a.genes.shape, device=a.genes.device) * (1 + 2*self.alpha) - self.alpha
        child_genes = a.genes + t * (b.genes - a.genes)
        return TensorIndividual(child_genes.to(a.genes.dtype))


# ============================================================================
# Mutation Layers
# ============================================================================

class GaussianMutationLayer(Layer):
    """Add gaussian noise to genes."""
    
    def __init__(self, rate: float = 0.1, std: float = 0.1):
        super().__init__()
        self.rate = rate
        self.std = std
    
    def execute(self, individuals: List[TensorIndividual]) -> List[TensorIndividual]:
        for ind in individuals:
            mask = torch.rand(ind.genes.shape, device=ind.genes.device) < self.rate
            noise = torch.randn(ind.genes.shape, device=ind.genes.device) * self.std
            ind.genes = (ind.genes + mask * noise).to(ind.genes.dtype)
        return individuals


class RandomMutationLayer(Layer):
    """Replace random genes with new random values."""
    
    def __init__(self, rate: float, low: float, high: float):
        super().__init__()
        self.rate = rate
        self.low = low
        self.high = high
    
    def execute(self, individuals: List[TensorIndividual]) -> List[TensorIndividual]:
        for ind in individuals:
            mask = torch.rand(ind.genes.shape, device=ind.genes.device) < self.rate
            new_values = random_tensor(ind.genes.shape, self.low, self.high, ind.genes.dtype, ind.genes.device)
            ind.genes = torch.where(mask, new_values, ind.genes)
        return individuals


class SwapMutationLayer(Layer):
    """Swap two random positions in each individual's tensor."""
    
    def execute(self, individuals: List[TensorIndividual]) -> List[TensorIndividual]:
        for ind in individuals:
            i, j = torch.randperm(ind.genes.shape[0])[:2]
            ind.genes[i], ind.genes[j] = ind.genes[j].clone(), ind.genes[i].clone()
        return individuals
