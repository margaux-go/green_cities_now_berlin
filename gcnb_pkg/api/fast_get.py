import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
#add import of processing function
from gcnb_pkg.preprocessing.encoding_pipeline import processing_pipe

app = FastAPI()

final_pipe = pickle.load(open('../../pickle/final_pipe.pkl', 'rb'))
model = pickle.load(open('../../pickle/model.pkl', 'rb'))

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
    usetype_bl: str,
    district: str,
    built_type: str,
    residents: float,
    air_pollut: str,
    thermal_st: str,
    social_sta: str,
    social_dyn: str,
    rent: float,
    unemp_bene: float,
    social_hou: float,
    hous_assoc: float,
    rent_durat: float,
    aparts_sol: float,
    subsidized: bool):


    input = pd.DataFrame(dict(
        usetype_bl=[usetype_bl],
        district=[district],
        built_type=[built_type],
        residents=[residents],
        air_pollut=[air_pollut],
        thermal_st=[thermal_st],
        social_sta=[social_sta],
        social_dyn=[social_dyn],
        rent=[rent],
        unemp_bene=[unemp_bene],
        social_hou=[social_hou],
        hous_assoc=[hous_assoc],
        rent_durat=[rent_durat],
        aparts_sol=[aparts_sol],
        subsidized=[subsidized]))


    preproc_X = final_pipe.transform(input)

    prediction = model.predict(preproc_X)

    return bool(prediction)


@app.get("/")
def root():
    return {'greeting': 'Hello'}
