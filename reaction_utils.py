from typing import List, Tuple
import pandas as pd

def read_reaction_file(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Reads a reaction file with SMARTS and reaction name columns (no header).
    Uses tab as separator to avoid conflicts with SMARTS patterns.
    
    Args:
        filepath: Path to the reaction file
        
    Returns:
        Tuple of (reaction_smarts_list, reaction_names_list)
    """
    try:
        # Read the file without header, using tab as separator
        df = pd.read_csv(filepath, header=None, names=['smarts', 'name'], sep='\t')
        return df['smarts'].tolist(), df['name'].tolist()
    except Exception as e:
        raise ValueError(f"Error reading reaction file: {str(e)}")

def validate_reaction_file(filepath: str) -> bool:
    """
    Validates that a reaction file has the correct format.
    Expects tab-separated values with SMARTS and reaction name columns.
    
    Args:
        filepath: Path to the reaction file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        df = pd.read_csv(filepath, header=None, names=['smarts', 'name'], sep='\t')
        # Check for empty file
        if len(df) == 0:
            return False
        # Check for missing values
        if df.isnull().any().any():
            return False
        # Check for empty strings
        if (df['smarts'].str.strip() == '').any() or (df['name'].str.strip() == '').any():
            return False
        return True
    except Exception:
        return False 