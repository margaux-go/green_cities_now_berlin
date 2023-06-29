# Climate Change Adaptation Analysis for Berlin
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

link to raw data files can be found here:
- GOOGLE CLOUD FOLDER LINK

The merged datasets included the following (Originally published in German):
- "Gebäudealter der Wohnbebauung" (Building Age)
- "Reale Nutzung der bebauten Flächen 2021" (Usage of built environment)
- "Gründächer - Block- und Blockteilflächen 2020" (Green roof on block level)
- "Gesamtindex Soziale Ungleichheit (Status/Dynamik-Index) 2021 (MSS)" (Social Inequality)
- "Umweltgerechtigkeit: Kernindikator Thermische Belastung 2021/2022" (Thermal Stress)
- "Luftbelastung 2021/2022" (Air Pollution)
- "Wohnatlas Berlin 2020" (Rent statistics)
- Green Roofing Subsidy programme postal code areas (As described in the official documentation: 'Förderrichtlinie zum Programm "GründachPLUS")

Before encoding and scaling of the data, the dataset was further cleaned (using the 'preprocess_dataframe' function in the data_cleaning.ipynb). This was necessary, due to unbuilt-areas, such as parks causing a larger number of missing values. Additionally, the choice was made to drop a number of features that did not seem to offer additional information beyond already existing, broader scale features (e.g. 'number of townhouses' vs 'residential' usetype). This pre-processing step can be easily adjusted to include more or fewer features.

In total, the final cleaned dataset consisted of 15 features, and the building block coordinates for 18,015 blocks in Berlin. A description of each variable with the label names can be found here:
- https://docs.google.com/spreadsheets/d/1Tsb1AojBWzbf_F6zUKElSugQRQkiqckQYKOMtTAgXZU/edit?usp=sharing

# Model
In search of the most efficient model for our case, we crossvalidated several models and based on

# Working with our code (to be updated)

Go to `https://github.com/{group}/newpkgname` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/newpkgname.git
cd newpkgname
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
newpkgname-run
```
