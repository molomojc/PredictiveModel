import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from DataCleanup import DataCleanup as dc
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler # StandardScaler for scaling numeric features


#TRAINING DATA PHASE
data = dc("train").cleanData() #called the DataCleanup class to clean the data
y = pd.to_numeric(data['VehicleSold'], errors='coerce')  # Convert to numeric, force bad values to NaN
y = y.fillna(0).astype(int)  # Replace NaNs and convert to int
X = data.drop(columns=['VehicleSold' , 'CustomerID', 'LeadID']) #features excluding the target variable 
# One-hot encoding for categorical variables

X_numeric = X[['CellPhoneNoLength', 'HourOfEnquiry', 'DayOfEnquiry', 'DTLeadCreated', 'DTLeadAllocated']]
#Add a difference column for the time difference between LeadCreated and LeadAllocated
X_numeric['LeadDelay'] = X['DTLeadAllocated'] - X['DTLeadCreated']
X_Categorical = X[['LeadType', 'LeadSource', 'CellPrefix', 'Domain', 'Dealer','Seek', 'InterestMake', 'InterestModel']]

General_Encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_array = General_Encoder.fit_transform(X_Categorical) #this is the encoded array


scaler = StandardScaler()
X_numeric_scaled = pd.DataFrame(scaler.fit_transform(X_numeric), columns=X_numeric.columns)


print("Encoded array shape:", encoded_array.shape)

encoded_cols = General_Encoder.get_feature_names_out()
X_cat_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=X.index)

X_full = pd.concat([X_numeric_scaled, X_cat_df], axis=1)

print(X_full.columns)
#Dala Train_split
#rus = RandomUnderSampler(random_state=42)
#X_resampled, y_resampled = rus.fit_resample(X_cat_df, y)

# Step 3: Split into train/val sets
X_train, X_val, y_train, y_val = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Train and evaluate
model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_resampled, y_resampled)

# Predict and evaluate
y_pred = model.predict(X_val)
print("Confusion matrix: ",confusion_matrix(y_val, y_pred))
print("classification Report: ",classification_report(y_val, y_pred))

'''
#NOTE: A Section for testing the model on unseen data
data_test = dc("test").cleanData("test") #called the DataCleanup class to clean the data
X_test = data_test.drop(columns=['CustomerID', 'LeadID']) #features excluding the target variable
tract categorical columns from test set, same as train
X_test_categorical = X_test[['LeadType', 'LeadSource', 'CellPrefix', 'Domain', 'Dealer', 'Seek', 'InterestMake', 'InterestModel']]

# Use the SAME encoder fitted on training data to transform test data
encoded_test_array = General_Encoder.transform(X_test_categorical)

# Create DataFrame for test encoded features with same columns as training
X_test_cat_df = pd.DataFrame(encoded_test_array, columns=encoded_cols, index=X_test.index)

# Make predictions on test data
test_predictions = model.predict(X_test_cat_df)

print("Test predictions:", test_predictions)
print("From the Train set:",y.value_counts(normalize=True))
print("From the Test set:", pd.Series(test_predictions).value_counts(normalize=True))
print(confusion_matrix(y, test_predictions))
print(classification_report(y, test_predictions))
'''







