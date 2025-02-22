from typing import List, Callable, Dict, Any
import numpy as np

def hamming_distance(p1: List[Any], p2: List[Any]) -> int:
    """
    Calculate Hamming distance between two permutations.
    (Number of positions where elements differ)
    
    Args:
        p1, p2: Permutations to compare
        
    Returns:
        Hamming distance
    """
    return sum(1 for x, y in zip(p1, p2) if x != y)

def displacement_distance(p1: List[int], p2: List[int]) -> int:
    """
    Calculate total displacement between two permutations.
    (Sum of |p1[i] - p2[i]| for all i)
    
    Args:
        p1, p2: Permutations to compare (must be integer permutations)
        
    Returns:
        Total displacement
    """
    return sum(abs(x - y) for x, y in zip(p1, p2))

def inversion_distance(p1: List[int], p2: List[int]) -> int:
    """
    Calculate inversion distance between two permutations.
    (Number of pairs (i,j) where i < j but p1[i] > p1[j] and p2[i] < p2[j] or vice versa)
    
    Args:
        p1, p2: Permutations to compare (must be integer permutations)
        
    Returns:
        Inversion distance
    """
    # Convert p2 to normalized form relative to p1
    normalized_p2 = [p1.index(x) for x in p2]
    
    # Count inversions in normalized_p2
    inversions = 0
    for i in range(len(normalized_p2)):
        for j in range(i+1, len(normalized_p2)):
            if normalized_p2[i] > normalized_p2[j]:
                inversions += 1
                
    return inversions

def cayley_distance(p1: List[int], p2: List[int]) -> int:
    """
    Calculate Cayley distance between two permutations.
    (Minimum number of transpositions to transform p1 to p2)
    
    Args:
        p1, p2: Permutations to compare (must be integer permutations)
        
    Returns:
        Cayley distance
    """
    # Convert p2 to normalized form relative to p1
    normalized_p2 = [p1.index(x) for x in p2]
    
    # Count cycles in the permutation
    visited = [False] * len(normalized_p2)
    cycle_count = 0
    
    for i in range(len(normalized_p2)):
        if not visited[i]:
            cycle_count += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = normalized_p2[j]
    
    # Distance is n - cycle_count
    return len(p1) - cycle_count

def get_distance_function(distance_type: str) -> Callable[[List[Any], List[Any]], float]:
    """
    Get a distance function by name.
    
    Args:
        distance_type: Type of distance ("hamming", "displacement", "inversion", "cayley")
        
    Returns:
        Distance function
    """
    distance_functions = {
        "hamming": hamming_distance,
        "displacement": displacement_distance,
        "inversion": inversion_distance,
        "cayley": cayley_distance
    }
    
    if distance_type not in distance_functions:
        raise ValueError(f"Unknown distance type: {distance_type}")
        
    return distance_functions[distance_type]