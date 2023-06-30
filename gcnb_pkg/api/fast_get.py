import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
#add import of processing function

app = FastAPI()

final_pipe = pickle.load(open('model/final_pipe.pkl', 'rb')) ###change code to use function in processing.py
model = pickle.load(open('model/model_15.pkl', 'rb'))  ### change to correct and final model pkl

# Allowing all middleware is optional, but good practice for dev purposes
### (what to we need this for?)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict_features(
    usetype_block: str,
    district: str,
    built_type: str,
    residents: float,
    air_pollution: str,
    thermal_stress: str,
    social_status: str,
    social_dyn: str,
    rent: float,
    unemp_benef: float,
    social_hou: float,
    hous_assoc: float,
    rent_duration: float,
    aparts_sold: float,
    subsidized: bool):


    input = pd.DataFrame(dict(
        usetype_block=[usetype_block],
        district=[district],
        built_type=[built_type],
        residents=[residents],
        air_pollution=[air_pollution],
        thermal_stress=[thermal_stress],
        social_status=[social_status],
        social_dyn=[social_dyn],
        rent=[rent],
        unemp_benef=[unemp_benef],
        social_hou=[social_hou],
        hous_assoc=[hous_assoc],
        rent_duration=[rent_duration],
        aparts_sold=[aparts_sold],
        subsidized=[subsidized]))


    preproc_X = final_pipe.transform(input)  ### change to use preprocessing function

    prediction = model.predict(preproc_X)

    return bool(prediction)


### can we delete this?
@app.get("/")
def root():
    return {'greeting': 'Hello'}
