import numpy as np
import random
from typing import List

from permutation_gym.envs.general_env import GeneralPermutationEnv
from permutation_gym.groups.custom import create_custom_group, create_from_cycles
from permutation_gym.utils.metrics import cayley_distance


def run_custom_group_experiment(n: int = 6, max_steps: int = 100, random_seed: int = 42):
    """
    Example of using a custom permutation group environment.
    
    This example shows:
    1. Creating a custom group with specified generators
    2. Using a custom distance metric (Cayley distance)
    3. Running episodes with random actions
    
    Args:
        n: Size of the permutation
        max_steps: Maximum steps per episode
        random_seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create custom generators using cycle notation
    # Define a set of generators using cycles
    cycles = [
        [0, 1, 2],  # 3-cycle rotating the first three elements
        [3, 4, 5],  # 3-cycle rotating the last three elements
        [0, 3],     # swap first and fourth elements
        [1, 4],     # swap second and fifth elements
    ]
    
    cycle_names = [
        "rotate_0_1_2", 
        "rotate_3_4_5", 
        "swap_0_3", 
        "swap_1_4"
    ]
    
    # Create generators from cycles
    generators, generator_names = create_from_cycles(n, cycles, cycle_names)
    
    # Create a random target permutation
    target = list(range(n))
    random.shuffle(target)
    
    # Create environment with custom cost function
    env = GeneralPermutationEnv(
        n=n,
        generators=generators,
        target_state=target,
        cost_fn=cayley_distance,  # Using Cayley distance
        max_steps=max_steps,
        group_type="custom",
        generator_names=generator_names
    )
    
    # Run episode
    print(f"Custom Group Example (n={n})")
    print(f"Initial state: Identity permutation {list(range(n))}")
    print(f"Target state: {target}")
    print(f"Using Cayley distance metric")
    
    observation, _ = env.reset()
    print(f"Starting state: {observation}")
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        # Take random action
        action = env.action_space.sample()
        
        # Execute step
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Print step information
        print(f"Step {step_count+1}:")
        print(f"  Applied generator: {info['generator_used']}")
        print(f"  New state: {observation}")
        print(f"  Reward: {reward}")
        print(f"  Cost (Cayley distance): {info['cost']}")
        print(f"  Total reward: {info['total_reward']}")
        
        # Check if done
        done = terminated or truncated
        step_count += 1
        
        if info["solved"]:
            print(f"Solved in {step_count} steps!")
            break
    
    if not info.get("solved", False):
        print(f"Failed to solve after {step_count} steps.")
        
    print(f"Final state: {observation}")
    print(f"Total reward: {info['total_reward']}")


if __name__ == "__main__":
    # Run with default parameters
    run_custom_group_experiment()