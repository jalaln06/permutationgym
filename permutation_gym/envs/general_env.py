import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import random

from .base_env import BasePermutationEnv

class GeneralPermutationEnv(BasePermutationEnv):
    """
    A gym environment for general permutation groups.
    
    This environment supports working with abstract permutation groups defined by
    their generators.
    """
    
    def __init__(self, 
                 n: int, 
                 generators: List[List[int]],
                 target_state: Optional[List[int]] = None,
                 cost_fn: Optional[Callable[[List[int], List[int]], float]] = None,
                 max_steps: int = 100,
                 group_type: str = "symmetric",
                 generator_names: Optional[List[str]] = None):
        """
        Initialize the environment for a general permutation group.
        
        Args:
            n: Number of elements in the permutation
            generators: List of generators defining the group
            target_state: Optional target permutation (if solving to a specific state)
            cost_fn: Function to calculate cost between permutations
            max_steps: Maximum steps before episode termination
            group_type: Type of group ("symmetric", "alternating", "cyclic", "dihedral", etc.)
            generator_names: Optional names for the generators (for interpretability)
        """
        super(GeneralPermutationEnv, self).__init__(n, generators, max_steps)
        
        self.group_type = group_type
        
        # Set generator names for interpretability
        if generator_names:
            if len(generator_names) != len(generators):
                raise ValueError("Number of generator names must match number of generators")
            self.generator_names = generator_names
        else:
            self.generator_names = [f"gen_{i}" for i in range(len(generators))]
        
        # Set target state
        if target_state is None:
            # Default target is the identity permutation
            self.target_state = list(range(n))
        else:
            if len(target_state) != n:
                raise ValueError(f"Target state must have length {n}")
            self.target_state = target_state
            
        # Set cost function
        if cost_fn is None:
            # Default cost is Hamming distance (number of elements not in correct position)
            self.cost_fn = lambda p1, p2: sum(1 for i, j in zip(p1, p2) if i != j)
        else:
            self.cost_fn = cost_fn
            
        # Initialize state
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Start with identity permutation (or randomize if specified in options)
        if options and options.get('randomize', False):
            self.current_state = list(range(self.n))
            random.shuffle(self.current_state)
        else:
            self.current_state = list(range(self.n))
            
        self.steps = 0
        self.total_reward = 0.0
        
        return np.array(self.current_state, dtype=np.int32), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Calculate current cost
        old_cost = self.cost_fn(self.current_state, self.target_state)
        
        # Apply the generator
        self.current_state = self.apply_generator(action, self.current_state)
        
        # Calculate new cost
        new_cost = self.cost_fn(self.current_state, self.target_state)
        
        # Calculate reward: -1 for each move, +100 if solved
        reward = 100.0 if new_cost == 0 else -1.0
        
        # Update total reward
        self.total_reward += reward
        
        # Update step count
        self.steps += 1
        
        # Check termination
        solved = (new_cost == 0)  # Zero cost means we reached the target
        terminated = solved or self.steps >= self.max_steps
        truncated = False
        
        return (
            np.array(self.current_state, dtype=np.int32),
            reward,
            terminated,
            truncated,
            {
                "cost": new_cost,
                "solved": solved,
                "steps": self.steps,
                "generator_used": self.generator_names[action],
                "total_reward": self.total_reward
            }
        )