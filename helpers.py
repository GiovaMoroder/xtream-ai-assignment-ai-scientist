# import data and remove non-sense points
import pandas as pd
import numpy as np

import torch 
from torch.utils.data import TensorDataset
import torch.nn.functional as F

#define numerical clarity variable
clarity_numerical_values = {
    'I1':   0,
    'SI2':  1,
    'SI1':  2,
    'VS2':  3,
    'VS1':  4,
    'VVS2': 5,
    'VVS1': 6,
    'IF':   7,
}


# Define numerical color variable
colors_numerical_values = {
    'J': 0,
    'I': 1,
    'H': 2,
    'G': 3,
    'F': 4,
    'E': 5,
    'D': 6
}

# Define cut numerical variables
cut_numerical_values = {
    'Fair': 0,
    'Good': 1,
    'Very Good': 2,
    'Premium': 3,
    'Ideal': 4
}



def group_indices(selected_color, selected_clarity):
    """
    Helper function to get the indices of the selected color and clarity levels
    """
    selected_color_idx = colors_numerical_values[selected_color]
    selected_clarity_idx = clarity_numerical_values[selected_clarity]
    # Increment clarity index
    # The data tensor contains the colors one hot vectors first 
    # and then the clarity one hot vectors 
    selected_clarity_idx += 7
    return selected_color_idx, selected_clarity_idx

def sorted_group_points(color_idx, clarity_idx, data):
    idx_group = (data.tensors[0][:,color_idx] == 1)*(data.tensors[0][:,clarity_idx] == 1)
    x = data.tensors[0][idx_group]
    y = data.tensors[1][idx_group]
    _, sorted_idx = torch.sort(x[:, -1])
    x = x[sorted_idx]
    y = y[sorted_idx]
    return x, y

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


def convert_to_torch_carat_color(df):
    """
    Helper function to convert the dataframe to torch tensors, perform the one-hot encoding of categorical variables compute the log of price and carat
    """

    # Change X
    X = df[['carat', 'color', 'clarity']].copy()

    # add numerical clarity and color variables
    X[['clarity_num', 'color_num']] = X.apply(
        lambda x: pd.Series({
            'clarity_num':  clarity_numerical_values[x['clarity']],
            'color_num':    colors_numerical_values[x['color']]
        }), axis = 1
    )
    del X['color']
    del X['clarity'] 

    X = torch.cat(
        [
            F.one_hot(torch.tensor(X['color_num'].to_numpy()), num_classes=7),
            F.one_hot(torch.tensor(X['clarity_num'].to_numpy()), num_classes=8),
            torch.tensor(X['carat'].to_numpy()).log().unsqueeze(1),
        ], axis = 1
    )
    X = X.float()

    # Preprocess the y variable if contained in the dataset
    if 'price' in df.columns:
        y = np.log(df['price'])
        y = y.to_numpy()
        y = torch.tensor(y, dtype=torch.float)

        # Create a TensorDataset from the preprocessed data
        data = TensorDataset(X, y)
    else: 
        data = X
    return data  

def convert_to_torch(df):
    """
    Helper function to convert the dataframe to torch tensors, perform the one-hot encoding of categorical variables compute the log of price and carat
    """

    # Change X
    X = df.copy()

    # Define numerical color variable
    # colors_numerical_values = {}
    # for i, l in enumerate(sorted(df.color.unique(), reverse=True)): 
    #     colors_numerical_values[l] = i

    # add numerical clarity, color, and cut variables
    X[['clarity_num', 'color_num', 'cut_num']] = X.apply(
        lambda x: pd.Series({
            'color_num':    colors_numerical_values[x['color']],
            'clarity_num':  clarity_numerical_values[x['clarity']],
            'cut_num':      cut_numerical_values[x['cut']] 
        }), axis = 1
    )
    del X['color']
    del X['clarity'] 
    del X['cut']

    # Combine all columns into a single tensor
    X = torch.cat(
        [
            F.one_hot(torch.tensor(X['color_num'].to_numpy()), num_classes=7),
            F.one_hot(torch.tensor(X['cut_num'].to_numpy()), num_classes=5),
            F.one_hot(torch.tensor(X['clarity_num'].to_numpy()), num_classes=8),
            torch.tensor(X['carat'].to_numpy()).log().unsqueeze(1),
            torch.tensor(X['depth'].to_numpy()).log().unsqueeze(1),
            torch.tensor(X['table'].to_numpy()).log().unsqueeze(1),
            torch.tensor(X['x'].to_numpy()).log().unsqueeze(1),
            torch.tensor(X['y'].to_numpy()).log().unsqueeze(1),
            torch.tensor(X['z'].to_numpy()).log().unsqueeze(1),
        ], axis = 1
    )
    X = X.float()

    # Preprocess the y variable if contained in the dataset
    if 'price' in df.columns:
        y = np.log(df['price'])
        y = y.to_numpy()
        y = torch.tensor(y, dtype=torch.float)

        # Create a TensorDataset from the preprocessed data
        data = TensorDataset(X, y)
    else: 
        data = X
    return data  

# def group_indices(selected_color, selected_clarity, preprocessor):
#     feature_names = preprocessor.get_feature_names_out()
#     color_idx = np.where(feature_names == f'onehot__color_{selected_color}')[0][0]
#     clarity_idx = np.where(feature_names == f'onehot__clarity_{selected_clarity}')[0][0]
#     return color_idx, clarity_idx

# Simple script to test the collection from torch tensor 
# based on color and clarity
# print(train_df.groupby(['color', 'clarity'])['carat'].count().sort_values(ascending = False)[:10])
# selected_color, selected_clarity = 'G', 'SI1'
# color_idx, clarity_idx = group_indices(selected_color, selected_clarity)
# x_train, y_train = sorted_group_points(color_idx, clarity_idx, training_data)
# print(x_train.shape)

# ((train_df['color'] == selected_color)&(train_df['clarity'] == selected_clarity)).sum()

def plot_compare_results_linear_models():
    raise NotImplementedError
