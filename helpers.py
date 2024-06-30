# import data and remove non-sense points
import pandas as pd
import numpy as np

def compute_perc(df, var):
    """
    Helper function to compute the percentage of data-points in each level for the categorical variable provided.
    
    Args:
        var (str): The name of the categorical variable.

    Yields:
        str: The level of the variable and its corresponding percentage.
    """
    
    for level in df[var].unique():
        # Compute the percentage of data-points in the current level
        percentage = round(len(df[df[var] == level])/len(df), 2)
        
        # Yield the level of the variable and its corresponding percentage
        yield level + ' ' + str(percentage)