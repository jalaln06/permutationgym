import ast
from typing import Dict, List, Tuple
import pandas as pd

def parse_santa_puzzle_file(puzzle_file: str) -> pd.DataFrame:
    """
    Parse the puzzle file from the Santa 2023 challenge.
    
    Args:
        puzzle_file: Path to puzzles.csv
        
    Returns:
        DataFrame containing puzzle information
    """
    return pd.read_csv(puzzle_file)

def parse_santa_puzzle_info_file(puzzle_info_file: str) -> Dict[str, Dict[str, List[int]]]:
    """
    Parse the puzzle info file from the Santa 2023 challenge.
    
    Args:
        puzzle_info_file: Path to puzzle_info.csv
        
    Returns:
        Dictionary mapping puzzle_type to allowed_moves
    """
    df = pd.read_csv(puzzle_info_file)
    
    # Create a dictionary mapping puzzle_type to allowed_moves
    puzzle_info_dict = {}
    for _, row in df.iterrows():
        puzzle_type = row['puzzle_type']
        allowed_moves = parse_allowed_moves(row['allowed_moves'])
        puzzle_info_dict[puzzle_type] = allowed_moves
        
    return puzzle_info_dict

def parse_allowed_moves(moves_str: str) -> Dict[str, List[int]]:
    """
    Parse the allowed_moves string from puzzle_info.csv.
    
    Args:
        moves_str: String representation of allowed moves
        
    Returns:
        Dictionary mapping move name to permutation
    """
    return ast.literal_eval(moves_str)

def parse_state(state_str: str) -> List[str]:
    """
    Parse state strings from the dataset.
    
    Args:
        state_str: State string from puzzle file
        
    Returns:
        List representation of the state
    """
    return state_str.split(';')