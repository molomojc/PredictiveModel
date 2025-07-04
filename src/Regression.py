from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Model import FeatureExtractor
from getData import DataFrame
import pandas as pd

# Extract columns and categories
model1 = FeatureExtractor('train')
Data1 = model1.extract_columns("train")
Target_variable = Data1['VehicleSold']
Data1 = Data1.drop(columns=['InFinanceProcessSystemApp', 'FinanceApplied', 'FinanceApproved', 'VehicleSold', 'OBSFullName',"OBSEmail"])
Data2 = model1.Extract_Category()
# Combine feature sets
X_Data = pd.concat([Data1, Data2], axis=1)
if 'DTLeadCreated' in X_Data.columns:
    X_Data['DTLeadCreated'] = pd.to_datetime(X_Data['DTLeadCreated']).astype(int) / 10**9 
if 'DTLeadAllocated' in X_Data.columns:
    X_Data['DTLeadAllocated'] = pd.to_datetime(X_Data['DTLeadAllocated']).astype(int) / 10**9  
    
#X_Data = X_Data.apply(pd.to_numeric, errors='coerce').fillna(0)
# Call the regression model
model = LogisticRegression()
# Fit the model
model.fit(X_Data, y= Target_variable)

model2 = FeatureExtractor('test')
data = model2.extract_columns("test") 
data = data.drop(columns=['OBSFullName',"OBSEmail"])
data2 = model2.Extract_Category()
X_DataTest = pd.concat([data, data2], axis=1)
if 'DTLeadCreated' in X_Data.columns:
    X_Data['DTLeadCreated'] = pd.to_datetime(X_Data['DTLeadCreated']).astype(int) / 10**9
if 'DTLeadAllocated' in X_Data.columns:
    X_Data['DTLeadAllocated'] = pd.to_datetime(X_Data['DTLeadAllocated']).astype(int) / 10**9
#X_Data = X_Data.apply(pd.to_numeric, errors='coerce').fillna(0)
# Get the test data

predictions = model.predict(X_Data)

accuracy = accuracy_score(Target_variable, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")

