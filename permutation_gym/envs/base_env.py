import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable

class BasePermutationEnv(gym.Env):
    """
    Base class for permutation group environments.
    """
    
    def __init__(self, 
                 n: int, 
                 generators: List[List[int]],
                 max_steps: int = 100):
        """
        Initialize the base permutation environment.
        
        Args:
            n: Number of elements in the permutation
            generators: List of generators defining the group
            max_steps: Maximum steps before episode termination
        """
        super(BasePermutationEnv, self).__init__()
        
        self.n = n
        self.generators = generators
        self.max_steps = max_steps
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(generators))
        self.observation_space = spaces.Box(0, n-1, shape=(n,), dtype=np.int32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment to initial state"""
        raise NotImplementedError("Subclasses must implement reset method")
    
    def step(self, action: int):
        """Execute one step in the environment"""
        raise NotImplementedError("Subclasses must implement step method")
    
    def apply_generator(self, gen_idx: int, state: List[int]) -> List[int]:
        """Apply a generator to the given state"""
        generator = self.generators[gen_idx]
        new_state = [state[i] for i in generator]
        return new_state