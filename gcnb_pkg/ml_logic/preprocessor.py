from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, make_union
import pickle

def clean_data(df_raw):
    """
    In this function we drop columns, rename columns from german to english,
    convert column types to the correct ones, drop NaNs for some columns and convert
    percentages to decimals.
    """
    # Drop columns to only keep the ones we want to use
    df = df_raw.drop(columns=['schl5','typ__fln','nutz','nutzung','ststrname','ststrnr','pl_id',
                      'index__ren','gruen20_m2','gruen20_p','typ__gr','bez','woz',
                      'status_num','bez_rent_a','plz','x1901_1910', 'x1911_1920',
                      'x1921_1930','x1931_1940','x1941_1950', 'x1951_1960', 'x1961_1970',
                      'x1971_1980','x1981_1990','x1991_2000', 'x2001_2010', 'x2011_2015',
                      'x_bis_1900','gint20_m2','gex20_m2','gint20_p','gex20_p','freistehen',
                      'doppelhaus','gereihtes','anderertyp','gruen16_m2','gruen16_p','diff_20_16',
                      'area_geb','anzahl_gru','anzahl_geb','flalle','ueberw_dek'])

    # Rename all the columns to their new value
    df.rename(columns = {'typklar__g':'usetype_block','bezirk':'district', 'woz_name':'built_type',
                         'ew2015':'residents', 'air_pollut':'air_pollution','aparts_sol':'aparts_sold',
                         'thermal_st':'thermal_stress','status_val':'social_status','dyn_val':'social_dyn',
                         'unemp_bene':'unemp_benef','city_owned':'hous_assoc', 'rent_durat':'rent_duration'}
                      , inplace = True)

    # Convert some column dtypes to their correct Dtype -> (From object to float32)
    df[['rent','unemp_benef','social_hou',
        'hous_assoc','rent_duration','aparts_sold']] = df[['rent','unemp_benef','social_hou','hous_assoc',
                                               'rent_duration','aparts_sold']].astype('float32')

    # Convert float64 columns into float32
    float64_cols = df.select_dtypes(include=['float64']).columns
    df[float64_cols] = df[float64_cols].astype('float32')

    # Convert int64 columns into bool. For 'green_roof' and 'subsidized' features
    int64_cols = df.select_dtypes(include=['int64']).columns
    df[int64_cols] = df[int64_cols].astype('bool')

    # Delete rows with NaN in 'built_type' column
    df = df[df['built_type'].isna()==False]

    # Delete rows where the rent is == 0. They give no useful information
    df = df[df['rent']!=0]

    # Get percentages from the columns
    df[['unemp_benef','social_hou','hous_assoc','rent_duration']] = df[['unemp_benef','social_hou','hous_assoc','rent_duration']] /100

    return df

def processing_pipe(X):

    #Categorical Encoder Piece of Pipeline
    ordinal_vals_1 = ['air_pollution']
    ordinal_vals_2 = ['social_status']
    ordinal_vals_3 = ['social_dyn']
    ordinal_vals_4 = ['thermal_stress']
    cat_vals = ['usetype_block','district','built_type']

    ordinal_encoder_1 = OrdinalEncoder(categories = [["gering","mittel","hoch"]])
    ordinal_encoder_2 = OrdinalEncoder(categories = [["sehr niedrig","niedrig","mittel","hoch"]])
    ordinal_encoder_3 = OrdinalEncoder(categories = [["negativ","stabil","positiv"]])
    ordinal_encoder_4 = OrdinalEncoder(categories = [["gering","mittel","hoch"]])

    cat_transformer = OneHotEncoder(drop='if_binary',handle_unknown='ignore',sparse_output=False)

    imputer = SimpleImputer(strategy="most_frequent")

    pipeline_1 = make_pipeline(imputer, ordinal_encoder_2)
    pipeline_2 = make_pipeline(imputer, ordinal_encoder_3)
    pipeline_3 = make_pipeline(imputer, ordinal_encoder_4)
    pipeline_4 = make_pipeline(imputer, ordinal_encoder_1)

    cat_pipe = ColumnTransformer([

        ('ordinal_transformer_1', pipeline_4, ordinal_vals_1),
        ('ordinal_transformer_2', pipeline_1, ordinal_vals_2),
        ('ordinal_transformer_3', pipeline_2, ordinal_vals_3),
        ('ordinal_transformer_4', pipeline_3, ordinal_vals_4),
        ('categorical_transformer', cat_transformer, cat_vals)
    ])

    #Numerical Scaler Piece of Pipeline
    num_transf_1 = make_pipeline(SimpleImputer(), StandardScaler())
    num_pipe = make_column_transformer((num_transf_1, make_column_selector(dtype_include=['float64','bool'])))

    #Combining Pieces to Form Final Pipeline
    combined_encoding = make_union(num_pipe, cat_pipe)

    final_pipe = ColumnTransformer([('combined_pipe', combined_encoding, X.columns)], remainder='passthrough').fit(X)

    #REMOVE IT HERE AT DUE TIME
    #processed_X = final_pipe.fit_transform(X)

    #Export final pipeline
    #if bool:
    #    final_pipe_file = "../pickle/final_pipe.pkl"
    #    pickle.dump(final_pipe, open(final_pipe_file, 'wb'))

    #return processed_X

    return final_pipe
