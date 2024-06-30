import pandas as pd 
import numpy as np 

from datetime import datetime
import json


class Pipeline(): 

    def __init__(self, new_data_train_perc, clean_data_mode, required_columns, data_path, models, model_selection_criterion):
        """
        Initialize the Pipeline object.

        Args:
            new_data_train_perc (float): The percentage of the new data to add to the training dataset.
            clean_data_mode (str): The mode of data cleaning to perform (currently only "delete").
            required_columns (list): The list of required columns in the data (predictors).
            data_path (dict): The dictionary containing the paths to the data and new data.
            models (list): The list of models to use.
            model_selection_criterion (str): The criterion to use for model selection ('date' for the latest model trained, 'accuracy' for the highest accuracy).
        """

        # Initialize the instance variables
        self.new_data_train_perc = new_data_train_perc  
        self.clean_data_mode = clean_data_mode          
        self.requried_columns = required_columns        
        self.new_data = None                            
        self.data_path = data_path                      
        self.models = models                            
        self.model_selection_criterion = model_selection_criterion 
        self.selected_model = None  

    def fetch_new_data(self):
        """
        Fetches new data from the path specified in self.data_path['new_data'] and checks that all required columns are present.
        """
        new_data = pd.read_csv(self.data_path['new_data'])

        # check all the required columns are present, discard all other columns
        try:
            assert all(col in new_data.columns for col in self.requried_columns), 'Not all required columns present'
        except AssertionError: 
            for col in self.requried_columns:
                if col not in new_data.columns:
                    print('Not present: ' + col)
            return None

        # store new data   
        new_data = {
            'train' : new_data,
            'test'  : None
        } 
        self.new_data = new_data
    
    def __clean_data_delete(self):
        """
        Cleans the new data fetched by removing all rows with nan values.
        """

        # access new_data
        new_data = self.new_data['train']

        # remove nan values or implement additional imputation techniques
        new_data = new_data[~new_data.isna().any(axis = 1)]
        
        # remove non-positive prices, table and depth
        new_data = new_data[new_data['price'] >0]
        new_data = new_data[new_data['table'] >0]
        new_data = new_data[new_data['depth'] >0]
        

        # remove non-positive lenghts  
        new_data = new_data[new_data['x'] >0]
        new_data = new_data[new_data['y'] >0]
        new_data = new_data[new_data['z'] >0]

        # add date to the new data
        new_data['date_added'] = pd.to_datetime('now')

        # shuffle the new data and create train test datasets
        idx = np.random.permutation(len(new_data))
        new_data = {
            'train':new_data.iloc[idx[:int(len(new_data)*self.new_data_train_perc)]],
            'test' : new_data.iloc[idx[int(len(new_data)*self.new_data_train_perc):]]
        }

        # store data 
        self.new_data = new_data
    
    def __clean_data_impute(self):
        """
        Cleans the new data fetched by imputing missing values.
        """
        raise NotImplementedError
    
    def __model_selection_date(self): 
        """
        Selects the latest model trained among the ones in the models_log.json file.
        The date and filenames of the selected models are stored in the self.selected_model.
        """

        # import json file
        with open('./pipeline/data_models/models_log.json') as json_file:
            model_log = json.load(json_file)
        
        models_selected = {}
        for m in self.models: 
            models_selected[m.model_name] = {
                'date' : "2000-01-15 12:00:00", # fake default date set to 2000 
                'filename': None
            }
        
        # update selected model based on model log
        for m in model_log: 
            model_date = datetime.strptime(m['date'], "%Y-%m-%d %H:%M:%S")
            current_date = datetime.strptime(models_selected[m['name']]['date'], "%Y-%m-%d %H:%M:%S")

            if model_date > current_date:
                models_selected[m['name']] = {
                    'date' : model_date.strftime("%Y-%m-%d %H:%M:%S"),
                    'filename' : m['filename']
                }

        # store the selected models
        self.models_selected = models_selected

    def __model_selection_r2(self): 

        raise NotImplementedError

    def clean_new_data(self): 
        """
        Cleans the new data fetched according to the modality specified in self.clean_data_mode.
        """

        # check if there is data to clean
        assert self.new_data is not None , 'No new data to clean'
        assert self.new_data['test'] is None, 'Data already cleaned'
        
        # clean the data
        if self.clean_data_mode == 'delete': 
            self.__clean_data_delete()
        if self.clean_data_mode == 'impute':
            self.__clean_data_impute()
    
    def save_new_data(self, verbose = 0):
        """
        Saves the new data in the main train and test datasets specified in self.data_path['data'].
        """
        # add the new data to the datasets
        def add_data(mode, verbose):

            data = pd.read_csv(self.data_path['data']+mode+'.csv')
            data = pd.concat([data, self.new_data[mode]], axis=0)
            data.to_csv(self.data_path['data']+mode+'.csv')

            # print data added
            if verbose > 0:
                print(len(self.new_data[mode]), 'new ' + mode + ' data added')
                print('Current ' + mode + ' size: ', len(data))
            
        add_data('train', verbose)
        add_data('test', verbose)

        # remove stored new data
        self.new_data = None

    def train_models(self, save_models = True): 
        """
        Trains all the models in self.models.
        """
        for m in self.models:
            m.train()
            m.evaluate()
            if save_models: m.save()
        
    def predict(self, df):
        """
        Performs the prediction on the dataset df using the models in self.models.
        """

        results = {}
        for m in self.models:
            results[m.model_name] = m.predict(df.copy())
        return results
    
    def select_model(self, verbose = False):
        """
        Selects the models in from the ones saved in models_log.json file according to the criterion in self.model_selection_criterion."""

        if self.model_selection_criterion == 'date':
            self.__model_selection_date()
        elif self.model_selection_criterion == 'r2': 
            self.__model_selection_r2()
        else: 
            print('Invalid selection criterion')
        
        if verbose:
            print('Models selected :', self.models_selected)
        
    def swap_models(self):
        """
        Imports the models in self.models_selected.
        """
        for m in self.models:
            m.swap_model(self.models_selected[m.model_name]['filename'])

        
    
