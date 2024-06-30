from models import LogLinearModel, NNModel
from pipeline import *
import pandas as pd

if __name__ == "__main__":

    # Initialize the pipeline
    pipeline = Pipeline(
        new_data_train_perc = 0.8,
        clean_data_mode = 'delete',
        required_columns = ['cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z'],
        data_path={
            'data': './pipeline/data_models/',
            'new_data': './pipeline/new_data/new_data.csv',
        },
        models = [
            LogLinearModel(
                model_name  = 'loglinear',
                model_path  = './pipeline/data_models/',
                data_path   = './pipeline/data_models/',
            ), 
            NNModel(
                model_name  = 'nn',
                model_path  = './pipeline/data_models/',
                data_path   = './pipeline/data_models/',
            )

        ],
        model_selection_criterion='date'
    )

    # Run pipeline functions

    pipeline.fetch_new_data() # fetch new data  
    pipeline.clean_new_data() # clean the new data
    pipeline.save_new_data(verbose = 1) # save the new data in the main train and test datasets
    pipeline.train_models(save_models = True) # train all the models specified and save them to file
    pipeline.select_model(verbose = True) # select models to deploy based on criterion
    pipeline.swap_models() # change the models in the pipeline importing the ones selected in pipeline.select()
    
    # import some data to make predictions using the pipeline
    d = pd.read_csv('./pipeline/new_data/new_data.csv')
    d.drop('price', axis = 1, inplace = True)
    d = d.iloc[:5]


    results = pipeline.predict(d)
    print(results)




    
    

    