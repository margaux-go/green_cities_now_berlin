import numpy as np
import pandas as pd
import geopandas as gpd
import pickle

from sklearn.compose import ColumnTransformer

def save_data(df: pd.DataFrame, root_path:str, suffix:str) -> None:
    """
    Add docstring here
    """
    df.to_csv(root_path + 'raw_data/project_data' + suffix+'.csv', sep=',', header = True)
    return None


def save_data_geo(df: gpd.GeoDataFrame, root_path:str, suffix:str) -> None:
    """
    Add docstring here
    """
    df.to_file(root_path + 'raw_data/project_data' + suffix+'.shp')
    return None


def save_pipeline(pipe: ColumnTransformer, root_path: str) -> None:
    """
    Add docstring here
    """
    pipe_file = root_path + 'pickle/final_pipe.pkl'
    pickle.dump(pipe, open(pipe_file, 'wb'))
    return None


def save_model(model, root_path:str) -> None:
    """
    Add docstring here
    """
    model_file = root_path + 'pickle/model.pkl'
    pickle.dump(model, open(model_file, 'wb'))
    return None


def load_data(root_path: str, suffix: str) -> pd.DataFrame:
    preprocessed_df = pd.read_csv(root_path + 'raw_data/project_data'+ suffix + '.csv', index_col = 0)
    return preprocessed_df


def load_model(root_path:str):
    model = pickle.load(open(root_path + 'pickle/model.pkl', 'rb'))
    return model
