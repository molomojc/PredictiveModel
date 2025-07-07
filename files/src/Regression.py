from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Model import FeatureExtractor
from sklearn.impute import SimpleImputer
import pandas as pd

# === TRAINING DATA ===
model1 = FeatureExtractor('train')

# Extract numeric and categorical
Data1 = model1.extract_columns()
Target_variable = Data1['VehicleSold']
Data1 = Data1.drop(columns=[
    'InFinanceProcessSystemApp', 'FinanceApplied', 'FinanceApproved', 
    'VehicleSold', 'OBSFullName', 'OBSEmail'
])

# Convert dates to numeric (Unix timestamp)
if 'DTLeadCreated' in Data1.columns:
    Data1['DTLeadCreated'] = pd.to_datetime(model1.data['DTLeadCreated'], errors='coerce').astype('int64') / 10**9
if 'DTLeadAllocated' in Data1.columns:
    Data1['DTLeadAllocated'] = pd.to_datetime(model1.data['DTLeadAllocated'], errors='coerce').astype('int64') / 10**9

# Encode categoricals
Data2 = model1.Extract_Category()
X_train = pd.concat([Data1, Data2], axis=1)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# Fit model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_imputed, Target_variable)

# === TESTING DATA ===
model2 = FeatureExtractor('test')
TestData1 = model2.extract_columns("test").drop(columns=['OBSFullName', 'OBSEmail'])

# Convert test date features using TEST dates
if 'DTLeadCreated' in TestData1.columns:
    TestData1['DTLeadCreated'] = pd.to_datetime(model2.data['DTLeadCreated'], errors='coerce').astype('int64') / 10**9
if 'DTLeadAllocated' in TestData1.columns:
    TestData1['DTLeadAllocated'] = pd.to_datetime(model2.data['DTLeadAllocated'], errors='coerce').astype('int64') / 10**9

# Encode test categoricals using TRAIN encoder
TestData2 = model2.Extract_Category(encoder=model1.encoder)
X_test = pd.concat([TestData1, TestData2], axis=1)

# Align columns (ensure test has same columns as train)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Impute missing values
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Predict on test data
predictions = model.predict(X_test_imputed)

# Evaluate on training data (optional, just for checking)
train_preds = model.predict(X_train_imputed)
train_acc = accuracy_score(Target_variable, train_preds)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
