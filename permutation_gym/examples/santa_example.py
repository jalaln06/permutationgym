import random
import pandas as pd
from typing import Dict

from permutation_gym.envs.santa_env import SantaPermutationEnv
from permutation_gym.utils.parsers import parse_santa_puzzle_file, parse_santa_puzzle_info_file, parse_allowed_moves, parse_state


def run_santa_puzzle_experiment(puzzle_id: int = 0, max_steps: int = 100, random_seed: int = 42):
    """
    Example of using the Santa 2023 puzzle environment.
    
    This example shows:
    1. Loading and parsing puzzle data
    2. Creating a Santa puzzle environment
    3. Running episodes with random actions
    
    Args:
        puzzle_id: ID of the puzzle to solve
        max_steps: Maximum steps per episode
        random_seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(random_seed)
    
    # Load puzzle data
    puzzles_df = parse_santa_puzzle_file("data/puzzles.csv")
    puzzle_info_dict = parse_santa_puzzle_info_file("data/puzzle_info.csv")
    
    # Get puzzle by ID
    puzzle = puzzles_df[puzzles_df["id"] == puzzle_id].iloc[0]
    
    # Parse puzzle data
    puzzle_type = puzzle["puzzle_type"]
    initial_state = parse_state(puzzle["initial_state"])
    solution_state = parse_state(puzzle["solution_state"])
    num_wildcards = int(puzzle["num_wildcards"])
    
    # Get allowed moves for this puzzle type
    allowed_moves = puzzle_info_dict[puzzle_type]
    
    # Create environment
    env = SantaPermutationEnv(
        puzzle_type=puzzle_type,
        initial_state=initial_state,
        solution_state=solution_state,
        allowed_moves=allowed_moves,
        num_wildcards=num_wildcards,
        max_steps=max_steps,
        reward_type="sparse"
    )
    
    # Run episode
    print(f"Santa 2023 Puzzle {puzzle_id} ({puzzle_type})")
    print(f"Number of elements: {len(initial_state)}")
    print(f"Number of wildcards: {num_wildcards}")
    print(f"Number of allowed moves: {len(allowed_moves)}")
    
    observation, _ = env.reset()
    print(f"Starting state encoded: {observation[:10]}...")  # Showing just the first 10 elements
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        # Take random action
        action = env.action_space.sample()
        
        # Execute step
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Print step information
        print(f"Step {step_count+1}:")
        print(f"  Applied move: {info['move_used']}")
        print(f"  Reward: {reward}")
        print(f"  Distance: {info['distance']:.4f}")
        print(f"  Total reward: {info['total_reward']}")
        
        # Check if done
        done = terminated or truncated
        step_count += 1
        
        if info["solved"]:
            print(f"Solved in {step_count} steps!")
            break
    
    if not info.get("solved", False):
        print(f"Failed to solve after {step_count} steps.")
        
    print(f"Final distance: {info['distance']:.4f}")
    print(f"Total reward: {info['total_reward']}")


if __name__ == "__main__":
    # Run with default parameters (puzzle 0)
    run_santa_puzzle_experiment()
    
    # Try another puzzle
    run_santa_puzzle_experiment(puzzle_id=1, max_steps=200)