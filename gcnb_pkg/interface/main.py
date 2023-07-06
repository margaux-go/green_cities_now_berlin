import pandas as pd
import geopandas as gpd

from gcnb_pkg.ml_logic.preprocessor import clean_data, processing_pipe
from gcnb_pkg.ml_logic.registry import save_data, save_data_geo, save_pipeline

def preprocess(root_path: str) -> None:
    """
    Add docstring here
    """
    print("Starting preprocessing 🤸\n")

    print('''\n Importing raw data.\n''')

    df_raw = gpd.read_file(root_path+'raw_data/project_data.shp')

    # We decide to call a block green when more than 15% of its area is covered
    # by green roofs
    green_thresshold = 15
    df_raw['green_roof'] = (df_raw['gruen20_p']>green_thresshold).astype(int)

    # cleaning and saving clean data
    df = clean_data(df_raw)
    print('''\n Saving clean data for visualization purposes. Warning is
          expected due to column name length. This warning is important since
          once clean data is loaded for visualization, column names will be
          shorter.⚠️ \n''')
    save_data_geo(df, root_path, '_clean')

    # separating target and attribute. Notice we drop geometry since it is not
    # an attribute
    X_clean = df.drop(columns=['green_roof', 'geometry'])
    y = df['green_roof']

    #fitting the preprocessing pipeline and saving it
    final_pipe = processing_pipe(X_clean)
    print('''\n Saving pipeline to be used by the API.\n''')
    save_pipeline(final_pipe, root_path)

    X = pd.DataFrame(final_pipe.transform(X_clean))
    print('''\n Saving preprocessed data.\n''')
    save_data(X, root_path, '_preprocessed')

    print("✅ preprocess() done \n")

if __name__ == '__main__':
    preprocess("../../")
