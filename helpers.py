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

def import_clean_data(): 
    df = pd.read_csv('./datasets/diamonds/diamonds.csv')

    # Retain on only positive prices
    df = df[df['price'] >0]
    df = df[df['x'] >0]
    df = df[df['y'] >0]
    df = df[df['z'] >0]
    return df

def rmedspe(y_pred, y_true):
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    error = (y_pred - y_true)/y_true
    return np.nanmedian(error**2)


def introduce_variables(df):
    df['log_price'] = np.log(df['price'])
    df['log_carat'] = np.log(df['carat'])

    # define log quantities
    df['log_price'] = np.log(df['price'])
    df['log_carat'] = np.log(df['carat'])

    # Define numerical color variable
    colors_numerical_values = {}
    for i, l in enumerate(sorted(df.color.unique(), reverse=True)): 
        colors_numerical_values[l] = i + 1

    #define numerical clarity variable
    clarity_numerical_values = {
        'I1':   1,
        'SI2':  2,
        'SI1':  3,
        'VS2':  4,
        'VS1':  5,
        'VVS2': 6,
        'VVS1': 7,
        'IF':   8,
    }

    # add numerical clarity and color variables
    df[['clarity_num', 'color_num']] = df.apply(
        lambda x: pd.Series({
            'clarity_num':  clarity_numerical_values[x['clarity']],
            'color_num':    colors_numerical_values[x['color']]
        }), axis = 1
    )
    return df