from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
import logging
from pydantic import BaseModel
from h11 import Request


from models import LogLinearModel, NNModel
from pipeline import *


class InputData(BaseModel):
    carat: List[float]
    cut: List[str]
    color: List[str]
    clarity: List[str]
    depth: List[float]
    table: List[float]
    x: List[float]
    y: List[float]
    z: List[float]

class OutputData(BaseModel):
    loglinear: List[float]
    nn: List[List[float]]

app = FastAPI()

# Initialize the ML pipeline
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

@app.get("/retrain")
async def train_endpoint():

    # Reinitialize the pipeline
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
    pipeline.fetch_new_data() # fetch new data  
    pipeline.clean_new_data() # clean the new data
    pipeline.save_new_data(verbose = 1) # save the new data in the main train and test datasets
    pipeline.train_models(save_models = True) # train all the models specified and save them to file
    return {"Status": 'Successfully retrained'}

@app.get("/select")
async def select_model():
    pipeline.select_model(verbose= True )
    pipeline.swap_models()
    return {"Status": 'Models successfully selected'}

@app.get("/available_models")
async def available_models():
    with open('./pipeline/data_models/models_log.json') as f:
        models = json.load(f)
    return {"models": models}


@app.post("/predict", response_model=OutputData)
async def process_data(input_data: InputData):
    try:
        input_data = input_data.dict()
        input_data = pd.DataFrame(input_data)
        print('MLAPI input data: ', input_data)

        # Perform the prediction
        results = pipeline.predict(input_data)
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")