import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from DataCleanup import DataCleanup as dc
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression

#TRAINING DATA PHASE
data = dc("train").cleanData() #called the DataCleanup class to clean the data
y = pd.to_numeric(data['VehicleSold'], errors='coerce')  # Convert to numeric, force bad values to NaN
y = y.fillna(0).astype(int)  # Replace NaNs and convert to int
X = data.drop(columns=['VehicleSold' , 'CustomerID', 'LeadID']) #features excluding the target variable 
# One-hot encoding for categorical variables

X_Categorical = X[['LeadType', 'LeadSource', 'CellPrefix', 'Domain', 'Dealer','Seek', 'InterestMake', 'InterestModel']]

General_Encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_array = General_Encoder.fit_transform(X_Categorical) #this is the encoded array

print("Encoded array shape:", encoded_array.shape)

encoded_cols = General_Encoder.get_feature_names_out()
X_cat_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=X.index)



model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_cat_df, y)

data_test = dc("test").cleanData("test") #called the DataCleanup class to clean the data
X_test = data_test.drop(columns=['CustomerID', 'LeadID']) #features excluding the target variable

# Extract categorical columns from test set, same as train
X_test_categorical = X_test[['LeadType', 'LeadSource', 'CellPrefix', 'Domain', 'Dealer', 'Seek', 'InterestMake', 'InterestModel']]

# Use the SAME encoder fitted on training data to transform test data
encoded_test_array = General_Encoder.transform(X_test_categorical)

# Create DataFrame for test encoded features with same columns as training
X_test_cat_df = pd.DataFrame(encoded_test_array, columns=encoded_cols, index=X_test.index)

# Make predictions on test data
test_predictions = model.predict(X_test_cat_df)

print("Test predictions:", test_predictions[:500])
print("From the Train set:",y.value_counts(normalize=True))
print("From the Test set:", pd.Series(test_predictions).value_counts(normalize=True))








