import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    # import the data
    df = pd.read_csv('./datasets/diamonds/diamonds.csv')

    # add to the existing dataset the date it was added
    df['date_added'] = pd.to_datetime('2022-12-25 12:34:56') #XMas time!

    # Remove the nans and non-positive prices, table, depths
    df = df[~df.isna().any(axis = 1)]
    df = df[df['price'] >0]
    df = df[df['table'] >0]
    df = df[df['depth'] >0]

    # remove non-positive lenghts
    df = df[df['x'] >0]
    df = df[df['y'] >0]
    df = df[df['z'] >0]
    
    # shuffle the dataset
    df.sample(frac=1).reset_index(drop=True)
    df.reset_index(drop=True, inplace=True)

    # create new data subset 
    new_data_perc = .1
    idx         = np.random.permutation(len(df))
    new_data    = df.iloc[idx[:int(len(df)*new_data_perc)]]
    df          = df.iloc[idx[int(len(df)*new_data_perc):]]

    # divide df into train and test
    train_perc = .8
    idx = np.random.permutation(len(df))
    train = df.iloc[idx[:int(len(df)*train_perc)]]
    test  = df.iloc[idx[int(len(df)*train_perc):]]

    # save the datasets in csv format
    os.makedirs ('./pipeline/data_models', exist_ok=True)
    train.to_csv('./pipeline/data_models/train.csv', index=False)

    os.makedirs('./pipeline/data_models', exist_ok=True)
    test.to_csv('./pipeline/data_models/test.csv', index=False)

    os.makedirs     ('./pipeline/new_data', exist_ok=True)
    new_data.to_csv ('./pipeline/new_data/new_data.csv', index=False)