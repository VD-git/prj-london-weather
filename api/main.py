from fastapi import FastAPI
import os
import pickle
from pydantic import BaseModel, Field
import pandas as pd

os.chdir('..')
root_dir = os.path.dirname(os.path.realpath('__file__'))

class PayloadItem(BaseModel):
    cloud_cover: float
    sunshine: float
    global_radiation: float
    max_temp: float
    min_temp: float
    precipitation: float
    pressure: float
    snow_depth: float

app = FastAPI()

@app.on_event("startup")
async def startup_event(): 
    global MODEL
    # Building path to be imported
    model_path = os.path.join(root_dir, 'output', 'model.pkl')

    if os.path.isfile(model_path):
        MODEL = pickle.load(open(model_path, "rb"))

@app.get("/")
async def say_hello():
    return {"greeting": "Hello Mate!"}

@app.post("/prediction/")
async def output(payload: PayloadItem):

    # Building path to be imported
    model_path = os.path.join(root_dir, 'output', 'model.pkl')
    
    json_file = {
        "cloud_cover": payload.cloud_cover,
        "sunshine": payload.sunshine,
        "global_radiation": payload.global_radiation,
        "max_temp": payload.max_temp,
        "min_temp": payload.min_temp,
        "precipitation": payload.precipitation,
        "pressure": payload.pressure,
        "snow_depth": payload.snow_depth,
    }

    # Creating DataFrame to run the model with
    data = pd.DataFrame([json_file])
    
    # Importing models
    if os.path.isfile(model_path):
        MODEL = pickle.load(open(model_path, "rb"))
    
    # Building the inference model
    prediction = MODEL.predict(data)

    return {"response": float(prediction)}

