from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pickle
import json
import os


app  = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)


with open('Fertilizer.pickle','rb') as f:
    __model = pickle.load(f)

@app.get("/")
async def root():
    return {"message":"Hello"}


class Fertilizer(BaseModel):
    Nitrogen:float
    Phosphorus:float
    Potassium:float
    Temperature:float
    Humidity:float
    Moisture:float
    Crop_type:str

crop = {
    "barley":0,
    "cotton":1,
    "groundnuts":2,
    "maize":3,
    "millets":4,
    "oilseeds":5,
    "paddy":6,
    "pulses":7,
    "sugarcane":8,
    "tobacco":9,
    "wheat":10
}



@app.post("/api/predict")
async def model(data:Fertilizer):
    data = data.dict()
    
    n = data['Nitrogen']
    p = data['Phosphorus']
    k = data['Potassium']
    t = data['Temperature']
    h = data['Humidity']
    m = data['Moisture']
    c = data['Crop_type']

    # Label encoding values

    c = crop[c]

    result = __model.predict([[t,h,m,c,n,k,p]])[0]
    
    data = {"fertilizer":result}
    value = json.dumps(data)
    return json.loads(value)


port  = os.environ.get('PORT',5000)

if '__name__' == '__main__':
    uvicorn.run(app,'127.0.0.1',port)







