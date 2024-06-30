import httpx
import asyncio

# IMPORTANT: before running this you should run initialize_pipeline_data.py at least once, to generate the necessary folders and data
# You should also run pipeline_main.py at least once, to generate the file pipeline/data_models/models_log.json and save the models to select from 

# Synthetic data to test the prediction api 
# The data should be in the exact format specified in the API
# The color, cut, and clarity should be one of the allowed values, otherwise the pipeline will throw an error and the API call will fail
input_data = {
    'carat': [1.0, 2.0],
    'cut': ['Ideal', 'Ideal'],
    'color': ['H', 'E'],
    'clarity': ['IF', 'IF'],
    'depth': [1.0, 11.0],
    'table': [1.0, 2.0],
    'x': [1.0, 2.0],
    'y': [1.0, 2.0],
    'z': [1.0, 2.0]
}

# General function to test the API
async def test(url):
    async with httpx.AsyncClient() as client:
        try:
            if 'predict' in url:
                response = await client.post(url, json = input_data)  # Ensure this is a GET request
            else:
                response = await client.get(url)  # Ensure this is a GET request
            response.raise_for_status()  # Raise an error for bad status
            result = response.json()  # Parse JSON response
            print("Response from predict API:", result)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.json()}")
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            

# URLs to test the API endpoints
ursl = [
    "http://localhost:8000/available_models",
    "http://localhost:8000/select",
    "http://localhost:8000/predict",
    "http://localhost:8000/retrain",
    "http://localhost:8000/predict",

]

for url in ursl:
    print(url)
    asyncio.run(test(url))
    print('\n\n\n')