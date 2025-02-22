import numpy as np
import random
from typing import List

from permutation_gym.envs.general_env import GeneralPermutationEnv
from permutation_gym.groups.symmetric import create_symmetric_group


def run_symmetric_group_experiment(n: int = 4, max_steps: int = 100, random_seed: int = 42):
    """
    Example of using the symmetric group environment.
    
    This example shows:
    1. Creating a symmetric group Sn environment
    2. Setting a random target permutation
    3. Running episodes with random actions
    
    Args:
        n: Size of the permutation
        max_steps: Maximum steps per episode
        random_seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create symmetric group generators
    generators, generator_names = create_symmetric_group(n)
    
    # Create random target permutation
    target = list(range(n))
    random.shuffle(target)
    
    # Create environment
    env = GeneralPermutationEnv(
        n=n,
        generators=generators,
        target_state=target,
        max_steps=max_steps,
        group_type="symmetric",
        generator_names=generator_names
    )
    
    # Run episode
    print(f"Symmetric Group S{n} Example")
    print(f"Initial state: Identity permutation {list(range(n))}")
    print(f"Target state: {target}")
    
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
        print(f"  Cost: {info['cost']}")
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
    run_symmetric_group_experiment()
    
    # Run with larger permutation
    run_symmetric_group_experiment(n=6, max_steps=200)