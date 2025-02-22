import numpy as np
import random
from typing import List

from permutation_gym.envs.general_env import GeneralPermutationEnv
from permutation_gym.groups.dihedral import create_dihedral_group


def run_dihedral_group_experiment(n: int = 5, max_steps: int = 100, random_seed: int = 42):
    """
    Example of using the dihedral group environment.
    
    This example shows:
    1. Creating a dihedral group Dn environment
    2. Setting a specific target permutation (reversed order)
    3. Running episodes with random actions
    
    Args:
        n: Size of the regular polygon (number of vertices)
        max_steps: Maximum steps per episode
        random_seed: Random seed for reproducibility
    """
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create dihedral group generators
    generators, generator_names = create_dihedral_group(n)
    
    # Create target permutation: reversed order
    target = list(range(n))
    target.reverse()
    
    # Create environment
    env = GeneralPermutationEnv(
        n=n,
        generators=generators,
        target_state=target,
        max_steps=max_steps,
        group_type="dihedral",
        generator_names=generator_names
    )
    
    # Run episode
    print(f"Dihedral Group D{n} Example")
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
    run_dihedral_group_experiment()
    
    # Try with different polygon sizes
    run_dihedral_group_experiment(n=6, max_steps=200)