import numpy as np
import pandas as pd
import geopandas as gpd
import pickle

from sklearn.compose import ColumnTransformer

def save_data(df: pd.DataFrame, root_path:str, suffix:str) -> None:
    """
    """
    df.to_csv(root_path+'raw_data/project_data'+suffix+'.csv', sep=',', header = False)
    return None

def save_data_geo(df: gpd.GeoDataFrame, root_path:str, suffix:str) -> None:
    """
    """
    df.to_file(root_path+'raw_data/project_data'+suffix+'.shp')
    return None

def save_pipeline(pipe: ColumnTransformer, root_path:str) -> None:
    """
    """
    pipe_file = root_path+"/pickle/final_pipe.pkl"
    pickle.dump(pipe, open(pipe_file, 'wb'))
