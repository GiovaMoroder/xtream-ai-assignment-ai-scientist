from tarfile import data_filter
from urllib import response
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from typing import List
from pydantic import BaseModel
import os
import pandas as pd
import httpx
import json

# Define the data class for typing
class Data(BaseModel):
    carat: List[float]
    cut: List[str]
    color: List[str]
    clarity: List[str]
    depth: List[float]
    table: List[float]
    x: List[float]
    y: List[float]
    z: List[float]

app = FastAPI()

# Serve static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Show the first 5 records in the dataset by default
df = pd.read_csv('./datasets/diamonds/diamonds.csv')
data = df[:5].to_dict('records')
new_data = None
predicted_data = None
retrain_flag = False

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", 
                                      {
                                          "request": request, 
                                          "data": data, 
                                          "new_data": new_data, 
                                          "predicted_data": None, 
                                          "retrain_flag": retrain_flag
                                          })

@app.post("/save", response_class=HTMLResponse)
async def save_new_data(request: Request):
    global new_data

    # change new data to pandas dataframe
    new_data_df = pd.DataFrame(new_data)

    # save along with additional data
    if os.path.exists('./pipeline/new_data/new_data.csv'):
        old_data_df = pd.read_csv('./pipeline/new_data/new_data.csv')
        print(len(old_data_df))
        new_data_df = pd.concat([old_data_df, new_data_df], axis=0)
        print(len(new_data_df))
        new_data_df.to_csv('./pipeline/new_data/new_data.csv', index=False)
    else: 
        new_data_df.to_csv('./pipeline/new_data/new_data.csv', index=False) 
    
    # reset new data
    new_data = None
        
    return templates.TemplateResponse("index.html", 
                                      {
                                          "request": request, 
                                          "data": data, 
                                          "new_data": new_data, 
                                          "predicted_data": None, 
                                          "retrain_flag": retrain_flag
                                          })

@app.post("/add", response_class=HTMLResponse)
async def add_data(request: Request,
                   carat: List[float] = Form(...),
                   cut: List[str] = Form(...),
                   color: List[str] = Form(...),
                   clarity: List[str] = Form(...),
                   depth: List[float] = Form(...),
                   table: List[float] = Form(...),
                   price: List[float] = Form(...),
                   x: List[float] = Form(...),
                   y: List[float] = Form(...),
                   z: List[float] = Form(...)):

    global new_data
    if new_data is None: new_data = [] 
    for carat_, cut_, color_, clarity_, depth_, table_, price_, x_, y_, z_ in zip(carat, cut, color, clarity, depth, table, price, x, y, z):
        new_data.append(
            {
                'carat': carat_,
                'cut': cut_,
                'color': color_,
                'clarity': clarity_,
                'depth': depth_,
                'table': table_,
                'price': price_,
                'x': x_,
                'y': y_,
                'z': z_,
            }
        )            
    
    return templates.TemplateResponse("index.html", 
                                      {
                                          "request": request, 
                                          "data": data, 
                                          "new_data": new_data, 
                                          "predicted_data": None, 
                                          "retrain_flag": retrain_flag
                                          })

@app.post("/retrain", response_class=HTMLResponse)
async def trigger_function(request: Request):
    # Retrain the models with the new data by calling the ML API
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/retrain", timeout=None)
        print("Response from retrain API:", response.json())  # Print the response to the console

    retrain_flag = True

    return templates.TemplateResponse("index.html", 
                                      {
                                          "request": request, 
                                          "data": data, 
                                          "new_data": new_data, 
                                          "predicted_data": None, 
                                          "retrain_flag": retrain_flag
                                          })

@app.post("/range", response_class=HTMLResponse)
async def get_range(request: Request, start_index: int = Form(...), end_index: int = Form(...)):
    data = df.iloc[start_index:end_index].to_dict('records') 
    return templates.TemplateResponse("index.html", {"request": request, "data": data, "new_data":new_data, "predicted_data": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict_value(
                        request: Request,
                        carat: List[float] = Form(...),
                        cut: List[str] = Form(...),
                        color: List[str] = Form(...),
                        clarity: List[str] = Form(...),
                        depth: List[float] = Form(...),
                        table: List[float] = Form(...),
                        x: List[float] = Form(...),
                        y: List[float] = Form(...),
                        z: List[float] = Form(...)
                        ):
    
    input_data= {
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z,
    }
    print('Input data:', input_data)

    url_predict = "http://localhost:8000/predict"  # Adjust the URL to match your server configuration
    url_select = "http://localhost:8000/select"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url_select, timeout=10)
            response.raise_for_status()  # Raise an error for bad status
            result = response.json()  # Parse JSON response
            print("Response from select API:", result)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.json()}")
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url_predict, json = input_data, timeout=10)
            response.raise_for_status()  # Raise an error for bad status
            result = response.json()  # Parse JSON response
            print("Response from predict API:", result)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.json()}")
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")

    # Convert input data from dict of lists to list of dicts
    input_data_ = []
    for i in range(len(input_data['carat'])):
        input_data_.append({
            'carat':    input_data['carat'][i],
            'cut':      input_data['cut'][i],
            'color':    input_data['color'][i],
            'clarity':  input_data['clarity'][i],
            'depth':    input_data['depth'][i],
            'table':    input_data['table'][i],
            'x':        input_data['x'][i],
            'y':        input_data['y'][i],
            'z':        input_data['z'][i],
        })
    print('Input data in list of dict format:', input_data_)
    print('Results:', result)
    print(type(input_data_))


    print(type(response))
    for i, (l, nnm, nne) in enumerate(zip(result['loglinear'], result['nn'][0], result['nn'][1])):
        input_data_[i]['loglinear'] = l
        input_data_[i]['nnm'] = nnm
        input_data_[i]['nne'] = nne
    result = input_data_
    print('Converted successfully')
    print(result)
    print(type(result))

    return templates.TemplateResponse("index.html", 
                                      {
                                          "request": request, 
                                          "data": data, 
                                          "new_data": new_data, 
                                          "predicted_data": result, 
                                          "retrain_flag": retrain_flag
                                          })

    