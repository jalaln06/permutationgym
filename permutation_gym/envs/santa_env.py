import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import ast

from .base_env import BasePermutationEnv

class SantaPermutationEnv(BasePermutationEnv):
    """
    A Gym environment for permutation group puzzles from Santa 2023 challenge.
    """
    
    def __init__(self, 
                 puzzle_type: str,
                 initial_state: List[str], 
                 solution_state: List[str],
                 allowed_moves: Dict[str, List[int]],
                 num_wildcards: int = 0,
                 max_steps: int = 100,
                 reward_type: str = "sparse"):
        """
        Initialize the environment.
        
        Args:
            puzzle_type: Type of puzzle (e.g., "cube_2/2/2")
            initial_state: Starting permutation
            solution_state: Target permutation
            allowed_moves: Dictionary of move name -> permutation mapping
            num_wildcards: Number of wildcards (0 if none)
            max_steps: Maximum steps before episode termination
            reward_type: How to calculate rewards ("distance", "improvement", "solved", "sparse")
        """
        # Convert allowed_moves dictionary to list of generators
        generators = list(allowed_moves.values())
        
        # Get size from initial state
        n = len(initial_state)
        
        super(SantaPermutationEnv, self).__init__(n, generators, max_steps)
        
        # Store puzzle information
        self.puzzle_type = puzzle_type
        self.initial_state = initial_state.copy()
        self.solution_state = solution_state.copy()
        self.allowed_moves = allowed_moves
        self.move_names = list(allowed_moves.keys())
        self.num_wildcards = num_wildcards
        self.reward_type = reward_type
        
        # For state representation, we encode the current permutation
        self.unique_elements = sorted(set(solution_state))
        self.element_to_idx = {elem: idx for idx, elem in enumerate(self.unique_elements)}
        self.idx_to_element = {idx: elem for idx, elem in enumerate(self.unique_elements)}
        
        # Initialize state
        self.reset()
        
    def state_to_array(self, state: List[str]) -> np.ndarray:
        """Convert state from string representation to integer array"""
        return np.array([self.element_to_idx[elem] for elem in state], dtype=np.int32)
        
    def array_to_state(self, array: np.ndarray) -> List[str]:
        """Convert state from integer array to string representation"""
        return [self.idx_to_element[idx] for idx in array]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_state = self.initial_state.copy()
        self.steps = 0
        self.total_reward = 0.0
        
        return self.state_to_array(self.current_state), {}
    
    def apply_move(self, move_idx: int) -> List[str]:
        """Apply a move (generator) to the current state"""
        # Get the generator from the allowed_moves dictionary
        move_name = self.move_names[move_idx]
        move_mapping = self.allowed_moves[move_name]
        
        # Apply the permutation
        new_state = [None] * len(self.current_state)
        for i, idx in enumerate(move_mapping):
            new_state[i] = self.current_state[idx]
        
        return new_state
    
    def calculate_distance(self, state: List[str]) -> float:
        """Calculate normalized distance between current state and solution"""
        if self.num_wildcards == 0:
            # Simple case: count mismatches
            mismatches = sum(1 for a, b in zip(state, self.solution_state) if a != b)
            return mismatches / len(state)
        else:
            # Complex case with wildcards (simplified implementation)
            mismatches = sum(1 for a, b in zip(state, self.solution_state) if a != b)
            return max(0, mismatches - self.num_wildcards) / len(state)
    
    def calculate_reward(self, old_distance: float, new_distance: float, done: bool) -> float:
        """Calculate reward based on the specified reward type"""
        if self.reward_type == "distance":
            # Negative distance to goal (closer is better)
            return -new_distance
        elif self.reward_type == "improvement":
            # Reward based on improvement
            return old_distance - new_distance
        elif self.reward_type == "solved":
            # Binary reward only when solved
            return 1.0 if done and new_distance == 0 else 0.0
        elif self.reward_type == "sparse":
            # -1 for each move, +100 if solved
            return 100.0 if done and new_distance == 0 else -1.0
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def is_solved(self, state: List[str]) -> bool:
        """Check if the puzzle is solved"""
        if self.num_wildcards == 0:
            return state == self.solution_state
        else:
            mismatches = sum(1 for a, b in zip(state, self.solution_state) if a != b)
            return mismatches <= self.num_wildcards
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Calculate current distance before move
        old_distance = self.calculate_distance(self.current_state)
        
        # Apply the selected move
        self.current_state = self.apply_move(action)
        
        # Calculate new distance after move
        new_distance = self.calculate_distance(self.current_state)
        
        # Check if puzzle is solved
        solved = self.is_solved(self.current_state)
        
        # Update step count
        self.steps += 1
        
        # Check termination conditions
        terminated = solved or self.steps >= self.max_steps
        truncated = False
        
        # Calculate reward
        reward = self.calculate_reward(old_distance, new_distance, solved)
        
        # Update total reward
        self.total_reward += reward
        
        # Return observation, reward, termination flag, truncation flag, and info
        return (
            self.state_to_array(self.current_state),
            reward,
            terminated,
            truncated,
            {
                "distance": new_distance, 
                "solved": solved, 
                "steps": self.steps, 
                "total_reward": self.total_reward,
                "move_used": self.move_names[action]
            }
        )
    
    @staticmethod
    def parse_moves(moves_str: str) -> Dict[str, List[int]]:
        """Parse the allowed_moves string from puzzle_info.csv"""
        # Convert string representation to dictionary
        moves_dict = ast.literal_eval(moves_str)
        return moves_dict
    
    @staticmethod
    def parse_state(state_str: str) -> List[str]:
        """Parse state strings from the dataset"""
        return state_str.split(';')