import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from DataCleanup import DataCleanup as dc

#TRAINING DATA PHASE
data = dc("train").cleanData() #called the DataCleanup class to clean the data
y = data['VehicleSold'] #target variable
X = data.drop(columns=['VehicleSold']) #features excluding the target variable
# One-hot encoding for categorical variables

