from typing import List, Dict, Tuple, Optional

def create_custom_group(generators: List[List[int]], 
                        generator_names: Optional[List[str]] = None) -> Tuple[List[List[int]], List[str]]:
    """
    Create a custom permutation group from specified generators.
    
    Args:
        generators: List of generator permutations
        generator_names: Optional list of names for the generators
        
    Returns:
        Tuple of (generators, generator_names)
    """
    if generator_names is None:
        generator_names = [f"gen_{i}" for i in range(len(generators))]
    
    if len(generator_names) != len(generators):
        raise ValueError("Number of generator names must match number of generators")
        
    return generators, generator_names


def create_from_cycles(n: int, cycles: List[List[int]], 
                       cycle_names: Optional[List[str]] = None) -> Tuple[List[List[int]], List[str]]:
    """
    Create generators from cycle notation.
    
    Args:
        n: Size of the permutation
        cycles: List of cycles, where each cycle is a list of indices
        cycle_names: Optional list of names for the cycles
        
    Returns:
        Tuple of (generators, generator_names)
    """
    generators = []
    
    for cycle in cycles:
        # Create identity permutation
        gen = list(range(n))
        
        # Apply cycle (i_0, i_1, ..., i_k) -> (i_0->i_1, i_1->i_2, ..., i_k->i_0)
        for i in range(len(cycle)-1):
            gen[cycle[i]] = cycle[i+1]
        gen[cycle[-1]] = cycle[0]
        
        generators.append(gen)
    
    if cycle_names is None:
        cycle_names = [f"cycle_{i}" for i in range(len(cycles))]
    
    if len(cycle_names) != len(cycles):
        raise ValueError("Number of cycle names must match number of cycles")
        
    return generators, cycle_names