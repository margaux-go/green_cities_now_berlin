import pandas as pd
import geopandas as gpd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

from gcnb_pkg.ml_logic.preprocessor import clean_data, processing_pipe
from gcnb_pkg.ml_logic.registry import save_data, save_data_geo, save_pipeline
from gcnb_pkg.ml_logic.registry import load_data, save_model, load_model

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
    X_clean = df.drop(columns=['green_roof', 'geometry']).reset_index(drop = True)
    y = df['green_roof'].reset_index(drop = True)

    #fitting the preprocessing pipeline and saving it
    final_pipe = processing_pipe(X_clean)
    print('''\n Saving pipeline to be used by the API.\n''')
    save_pipeline(final_pipe, root_path)

    #transforming data and merging target values back to dataframe
    X = pd.DataFrame(final_pipe.transform(X_clean))
    preprocessed_df = X.merge(y, left_index = True, right_index = True)
    print('''\n Saving preprocessed data.\n''')
    save_data(preprocessed_df, root_path, '_preprocessed')

    print("✅ preprocess() done \n")

def initialize_train_model(root_path: str) -> None:
    """
    Add doc string here
    """

    print("\nInitializing and training model (Gradient Boosting Classifier) 🤸\n")

    print('''\n Importing preprocessed data.\n''')
    preprocessed_df = load_data(root_path, '_preprocessed')
    X = preprocessed_df.drop(columns = 'green_roof')
    y = preprocessed_df['green_roof']

    # defining model parameters
    model_params = {'ccp_alpha': 0.0,
    'criterion': 'squared_error',
    'init': None,
    'learning_rate': 0.09,
    'loss': 'exponential',
    'max_depth': 6,
    'max_features': None,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 6,
    'min_samples_split': 100,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 100,
    'n_iter_no_change': None,
    'random_state': 42,
    'subsample': 1.0,
    'tol': 0.0001,
    'validation_fraction': 0.1,
    'verbose': 0,
    'warm_start': False}

    model = GradientBoostingClassifier(**model_params).fit(X, y)

    print('\n Saving trained model. \n')
    save_model(model, root_path)

    print("✅ initialize_train_model() done \n")
    return None

def evaluate_model(root_path) -> None:
    """
    Add docstring here
    """
    print("\n Model Evaluation 🤸:\n")

    model = load_model(root_path)
    preprocessed_df = load_data(root_path, '_preprocessed')
    X = preprocessed_df.drop(columns = 'green_roof')
    y = preprocessed_df['green_roof']

    y_pred_proba = model.predict_proba(X)[:,1]

    # picking a threshold based on roc curve analysis
    threshold = 0.024697

    def custom_predict_proba(y_pred, threshold):
        """
        Function applies different threshold for final classification.
        For scores above the threshold classifiers returns 1.
        For scores under the threshold classifiers returns relative score w.r.t threshold.
        TODO: vectorization
        """
        predictions = []

        for pred in y_pred:
            if pred >= threshold:
                predictions.append(1)
            else:
                predictions.append(round(pred/threshold,3))
        return predictions

    # Computes scores w.r.t to threshold
    y_pred_threshold =  custom_predict_proba(y_pred_proba, threshold)
    y_pred = y_pred_proba>threshold

    cm = metrics.confusion_matrix(y, y_pred)
    tpr = cm[1][1]/(cm[1][1]+cm[1][0])
    fpr = cm[0][1]/(cm[0][0]+cm[0][1])

    print('Confusion Matrix:\n', cm)

    print('\ntpr is:',tpr)
    print('\nfpr is:',fpr)

    print(f"""This means that the model is capable of predicting {100 * tpr}% of the of the green roofs
    and { 100 * (1 - fpr)}% of the non-green roofs correctly.
    """)

    print("✅ evaluate_model() done \n")
    return None


if __name__ == '__main__':
    root_path = "../../"
    preprocess(root_path)
    initialize_train_model(root_path)
    evaluate_model(root_path)
