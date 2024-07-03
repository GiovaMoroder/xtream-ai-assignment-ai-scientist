import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from statsmodels.formula.api import ols

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from metrics import Metrics

import datetime
import json
import os
import pickle

class BaseModel(): 
    def __init__(self, model_name, data_path, model_path):
        """
        Initialize the BaseModel. All ML models in the pipeline should inherit from this class.

        Args:
            model_name (str): The name of the model.
            data_path (str): The path to the data (both train and test should be in this path).
            model_path (str): The path to store the model and the log of the trained models.
        """
        # Set the model name, data patha dand model path
        self.model_name = model_name
        self.data_path  = data_path
        self.model_path = model_path

    # Method to train the ML model
    def train(self):
        raise NotImplementedError

    # Method to predict using the trained ML model
    def predict(self):
        raise NotImplementedError
    
    # Method to evaluate the ML model using the specified metrics
    def evaluate(self):
        raise NotImplementedError

    # Method to save the ML model
    def save(self):
        raise NotImplementedError
    
    def __find_max_num(self):
        """
        Helper function that finds the number of files with the same model name in the model_path
        """
        max_num = 0

        for filename in os.listdir(self.model_path):
            if filename.startswith(self.model_name + '_') and filename.endswith('.pkl'):
                num = int(filename.split('_')[-1].split('.')[0])
                max_num = max(max_num, num)

        return max_num
    
    def __save_to_file(self, model_log):
        """
        Helper function that saves the model and model_log (containing the metrics of the model and the date of creation) to a file in model_path
        """
        # Find number of files with the same model name
        max_num = self.__find_max_num()

        # If the model is a torch object, save using torch.save
        if isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), f'{self.model_path}{self.model_name}_{max_num+1}.pth')
            model_log['filename'] = f'{self.model_path}{self.model_name}_{max_num+1}.pth'
        
        # Otherwise, save using pickle
        else:
            with open(
                f'{self.model_path}{self.model_name}_{max_num+1}.pkl', 'wb') as f:
                pickle.dump(self.model, f)
                model_log['filename'] = f'{self.model_path}{self.model_name}_{max_num+1}.pkl'

        # Save log to file
        if not os.path.exists(self.model_path+'models_log.json'):
            with open(self.model_path+'models_log.json', 'w') as f:
                json.dump([model_log], f)
        else:
            with open(self.model_path+'models_log.json', 'r') as f:
                models_log = json.load(f)
            models_log.append(model_log)
            with open(self.model_path+'models_log.json', 'w') as f:
                json.dump(models_log, f)


class LogLinearModel(BaseModel):
    def __init__(self, model_name, model_path, data_path ,formula = None):
        """
        Initialize a LogLinearModel object extending the BaseModel class.

        Args:
            model_name (str): Name of the model
            model_path (str): Path to save the model
            data_path (str): Path to load the data
            formula (str, optional): The formula to use for the linear model. Defaults to 'log_price ~ log_carat + clarity + color'.
        """
        
        # Call the constructor of the parent class
        super().__init__(model_name, data_path, model_path)

        # Set the formula for the linear model
        if formula is None:
            formula = 'log_price ~ log_carat + clarity + color'
        self.formula = formula

        # Initialize the data
        self.__init_data()

        # Initialize the model
        self.__init_model()

        # Specify the metrics to benchmark the model
        self.metrics_to_run = ['r2', 'r2_adj']

        # Initialize the dictionary of updated metrics
        self.model_metrics = None 

        # Set the data path
        self.data_path = data_path
    
    def __init_data(self): 
        """
        Helper function to import the data and apply the transformations needed
        """
        # Initialize the data dictionary
        self.data = {
            'train' : pd.read_csv(self.data_path + 'train.csv'),
            'test'  : pd.read_csv(self.data_path + 'test.csv')
        }

        # Apply log transformation to carat and price columns
        for n, d in self.data.items(): 
            d['log_carat'] = np.log(d['carat'])
            d['log_price'] = np.log(d['price'])

    def __init_model(self):
        """
        Helper function to initialize the  model
        """
        self.model = ols(formula=self.formula, data = self.data['train'])

    def train(self):
        """
        Train the model using the statsmodels library
        """
        self.model = self.model.fit() # huge pain: for some reason ols returns a model when  you call fit()
    
    def evaluate(self): 
        """
        Evaluate the model by running the specified metrics on the test and train data.
        The metrics are stored in the model_metrics dictionary, with keys of the form '{metric_name}-{metric_val}'.
        """

        # Initialize metrics object passing the metrics to run, and any additional arguments for the metrics
        metrics = Metrics(
            self.metrics_to_run, 
            args = {'n_features' : len(self.model.params)})

        # Dictionary to store the metrics for the model
        model_metrics = {}

        # Run metrics for each data set (train and test)
        for data_n, d in self.data.items(): 
            for met_n, met_v in metrics.run(self.model.predict(d), d['log_price']):
                model_metrics[data_n + '-' + met_n] = met_v
        
        # Store the metrics 
        self.model_metrics = model_metrics

    def predict(self, df):
        df['log_carat'] = np.log(df['carat'])
        return self.model.predict(df)

    def save(self):
        """
        Save the model to a file and the most important metrics to a json log file
        Both are saved to the directory specified by self.model_path.
        """

        # Compute the log file
        model_log = {
            'name'      : self.model_name,
            'formula'   : self.formula,
            'metrics'   : self.model_metrics,
            'date'      : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        # Save the model and the log file using parent helper function
        self._BaseModel__save_to_file(model_log)

    def swap_model(self, saved_model_filename):
        with open(saved_model_filename, 'rb') as f:
            self.model = pickle.load(f)


class NNModel(BaseModel):
    def __init__(self, model_name, model_path, data_path, hidden_dim=None, **kwargs):
        """
        Initialize a NNModel object extending the BaseModel class.

        Args:
            model_name (str): Name of the model
            model_path (str): Path to save the model
            data_path (str): Path to load the data
            hidden_dim (int, optional): Dimension of the hidden layer in the neural network. Defaults to 100.
        """

        # Call the constructor of the parent class
        super().__init__(model_name, data_path, model_path)

        # Initialize the hidden dimension
        self.hidden_dim = hidden_dim if hidden_dim is not None else 100

        # Initialize the number of features
        self.n_features = 16 # MTODO: make it more general
        self.preprocessor = None

        # Initialize the data
        self.__init_data()

        # Initialize the model
        self.__init_model()

        # Initialize the metrics to benchmark the model
        self.metrics_to_run = ['r2', 'r2_adj']

        # Dictionary to store the metrics for the model
        self.model_metrics = None 

        # Set the data path
        self.data_path = data_path

    def __init_data(self): 
        """
        Initialize the data for training and testing.
        """
        
        # Load and store the data
        self.data = {
            'train' : pd.read_csv(self.data_path + 'train.csv'),  
            'test'  : pd.read_csv(self.data_path + 'test.csv')    
        }

        # Preprocess the data
        self.data['train']  = self.preprocess_data(self.data['train'])  
        self.data['test']   = self.preprocess_data(self.data['test'])   

    def preprocess_data(self, df):
        # Change X
        X = df[['carat', 'color', 'clarity']].copy()

        # Define numerical color variable
        colors_numerical_values = {}
        for i, l in enumerate(sorted(df.color.unique(), reverse=True)): 
            colors_numerical_values[l] = i

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

        # Store the number of features
        self.n_features = X.shape[1]

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

    def __init_model(self):
        """
        Initialize the neural network model.
        The model consists of three fully connected layers that estimate the means and log standard deviations of the response variable given some input data
        """
        class Model(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super(Model, self).__init__()

                # Define nn layers
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, 1)
                self.fc3 = nn.Linear(hidden_dim, 1)
        
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                mean = self.fc2(x)
                log_std = self.fc3(x)
                return mean, log_std

        # Initialize the model
        self.model = Model(self.n_features, self.hidden_dim)

    def train(self, batch_size = 100, epochs = 6, lr = 1e-1, verbose = 5):
        # TODO: currently not using data loaders

        self.model.train()
        train_loader = DataLoader(self.data['train'], batch_size=batch_size, shuffle = True)
        test_loader  = DataLoader(self.data['train'], batch_size=batch_size, shuffle = False)

        #initializer optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # define custom loss function
        def criterion(y_true, y_pred, log_std): 
            loss =  ((y_pred - y_true)/log_std.exp())**2 + 2*log_std
            return loss.mean()
        
        # Training loop
        log_loss = list()
        if verbose > 0: print('NN training... :, ', self.model.training)
        for e in range(epochs): 
            optimizer.zero_grad()
            y_pred, log_std = self.model(self.data['train'].tensors[0])
            # print(output.shape)
            loss = criterion(
                y_pred  = y_pred.squeeze(), 
                y_true  = self.data['train'].tensors[1].squeeze(), 
                log_std = log_std.squeeze())
            loss.backward()
            optimizer.step() 

            if verbose > 0: 
                if e % (epochs//verbose) == 0: 
                    print(f'Epoch {e+1}/{epochs}, Loss: {loss.item()}')
            log_loss.append(loss.item()) 

    def evaluate(self): 
        """
        Evaluate the model by running the specified metrics on the test and train data.
        The metrics are stored in the model_metrics dictionary, with keys of the form '{metric_name}-{metric_val}'.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Initialize metrics object
        metrics = Metrics(
            self.metrics_to_run, 
            args = {'n_features' : self.data['test'].tensors[0].shape[1]})

        # Run the metrics
        model_metrics = {}
        for data_n, d in self.data.items(): 
            for met_n, met_v in metrics.run(
                    y_pred = self.model(d.tensors[0])[0].detach(), 
                    y_true = d.tensors[1].detach()
                ):
                model_metrics[data_n + '-' + met_n] = met_v
        
        self.model_metrics = model_metrics

    def predict(self, df):
        # df = df[['carat', 'color', 'clarity']] # retain only  relevant columns
        data = self.preprocess_data(df[['carat', 'color', 'clarity']])
        mean, log_std = self.model(data)
        mean, log_std = mean.detach().numpy(), log_std.detach().numpy()
        return mean, log_std
        

    def save(self):
        """
        Save the model to a file and the most important metrics to a json log file
        Both are saved to the directory specified by self.model_path.
        """
        model_log = {
            'name'      : self.model_name,
            'metrics'   : self.model_metrics,
            'date'      : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        }
        self._BaseModel__save_to_file(model_log)

    def swap_model(self, saved_model_filename):
        # swap the current model for one of the saved mode
        self.model.load_state_dict(torch.load(saved_model_filename))
        self.model.eval()
   
        