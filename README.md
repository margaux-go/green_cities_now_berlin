# Climate Change Adaptation Analysis for Berlin 🌳
This repo documents the code used in the Le Wagon final project 'Green_Cities_Now'. Goal of the project was to
train a classification model on building block data from the Berlin Open Data Portal, to predict the probability
of climate change adaptation measures being implemented.

What can be found in this repo:
- Preprocessing pipeline specific to the dataset used
- Notebooks showcasing model selection and training
- Pickle file with the trained model
- API for a prediction feature in the front-end application

Link to front-end:
- https://green-cities-now.streamlit.app/

The code to the front-end application is stored in the following repo:
- https://github.com/carvclauson/green_cities_now_frontend/


# Data
The data used in this project to train the classification model consists of several datasets
from the Berlin Open Data Portal that were merged at building block scale.

Link to raw data files can be found here (download into raw_data folder):
- https://drive.google.com/drive/folders/1rdh1HcbbfJZ5bLnFTM1krVcSbnqD6k-a?usp=sharing

The merged datasets included the following (Originally published in German):
- "Gebäudealter der Wohnbebauung" (Building Age)
- "Reale Nutzung der bebauten Flächen 2021" (Usage of built environment)
- "Gründächer - Block- und Blockteilflächen 2020" (Green roof on block level)
- "Gesamtindex Soziale Ungleichheit (Status/Dynamik-Index) 2021 (MSS)" (Social Inequality)
- "Umweltgerechtigkeit: Kernindikator Thermische Belastung 2021/2022" (Thermal Stress)
- "Luftbelastung 2021/2022" (Air Pollution)
- "Wohnatlas Berlin 2020" (Rent statistics)
- Green Roofing Subsidy programme postal code areas (As described in the official documentation: 'Förderrichtlinie zum Programm "GründachPLUS")

Before encoding and scaling of the data, the dataset was further cleaned (using the 'preprocess_dataframe' function in the data_cleaning.ipynb). This was necessary, due to unbuilt-areas, such as parks, causing a larger number of missing values. Additionally, the choice was made to drop a number of features that did not seem to offer additional information beyond already existing, broader scale features (e.g. 'number of townhouses' vs 'residential' usetype). This pre-processing step can be easily adjusted to include more or fewer features.

In total, the final cleaned dataset consisted of 15 features, and the building block coordinates for 18,015 blocks in Berlin. Encoding of features resulted in a total number of 75 features to be used in model-training. A description of each variable with the label names can be found here:
- https://docs.google.com/spreadsheets/d/1Tsb1AojBWzbf_F6zUKElSugQRQkiqckQYKOMtTAgXZU/edit?usp=sharing

The target variable 'green_roof', is defined by any building block, where more than 15% of the building footprint area is greened.

# Model
In search of the most efficient model for our case, we cross-validated several models. Based on the auc-score, we selected and tuned a GradientBoostingClassifier. The process of selection and training can be found in the 'model_selection' and 'model' notebooks. The final model is exported as a pickle file, to be found in the pickle folder.

Due to the dataset being quite unbalanced, a classification thresshold of 0.02 was implemented. Resulting in the final model being capable of predicting 87.94 % of the of the green roofs and 89.40% of the non-green roofs correctly.

# Instructions
The following sequence of steps should setup the code and illustrate its functionalities:
- Clone the repository with *git clone git@github.com:margaux-go/green_cities_now_berlin.git*
- At the folder green_cities_now_berlin create a subfolder raw_data with *mkdir raw_data* and download the raw data to this subfolder from the first link under the 'Data' section. Furthermore create a subfolder 'pickle' in which preprocess pipeline and trained model will be saved.
- We suggest the creation of a virtual environment for the project to avoid dependency issues. At the folder green_cities_now_berlin
run *pyenv virtualenv <env_name>* followed by *pyenv local <env_name>*
- At the folder green_cities_now_berlin, run *make install* to install necessary packages.
- At the folder green_cities_now_berlin, run *make run_preprocess* to preprocess the raw_data. It will clean the raw_data
and export the clean data for visualization purposes. Furthermore, it fits a preprocessing pipeline to the clean data,
and exports to be used by the API.
- At the folder green_cities_now_berlin, run *make run_train* to train the model and export it to be used by the API.
- At the folder green_cities_now_berlin, run *make run_evaluate* to see model evaluation.
- To test the API locally, first run *uvicorn fast_get:app --reload* at the folder api and then run the cells in the api_local_test.ipynb Jupyter notebook.
- The Jupyter notebook interface_explore.ipynb presents a break down of the gcnb package and visualization of the data.
- We also offer a docker image for containerization.

# Questions?

If you have any questions, feel free to contact us on linkedin:
- Margaux: https://www.linkedin.com/in/margaux-huth/
- Raquel: https://www.linkedin.com/in/raquelbrasileiro/
- Juanes: https://www.linkedin.com/in/juan-esteban-hoyos-g/
- Clauson: https://www.linkedin.com/in/clauson-da-silva/
