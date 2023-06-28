
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline, make_union




#Categorical Value Encoder Pipeline

ordinal_vals_1 = ['air_pollut']
ordinal_vals_2 = ['social_sta']
ordinal_vals_3 = ['social_dyn']
ordinal_vals_4 = ['thermal_st']
cat_vals = ['usetype_bl','district','built_type']

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

# to be added: numerical and pipeline merge
