'''
2nd Model with Random Forest Classifier
This model uses the Random Forest Classifier to predict the target variable.
'''
import pandas as pd
from DataCleanup import DataCleanup as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

# TRAINING DATA PHASE
data = dc("train").cleanData()  # called the DataCleanup class to clean the data 
y = pd.to_numeric(data['VehicleSold'], errors='coerce')  # Convert to numeric, force bad values to NaN
y = y.fillna(0).astype(int)  # Replace NaNs and convert to
X = data.drop(columns=['VehicleSold' , 'CustomerID', 'LeadID']) #features excluding the target variable 

X_numeric = X[['CellPhoneNoLength', 'HourOfEnquiry', 'DayOfEnquiry', 'DTLeadCreated', 'DTLeadAllocated']]
# Add a difference column for the time difference between LeadCreated and LeadAllocated 
#X_numeric['LeadDelay'] = X['DTLeadAllocated'] - X['DTLeadCreated']
X_Categorical = X[['LeadType', 'LeadSource', 'CellPrefix', 'Domain', 'Dealer','Seek', 'InterestMake', 'InterestModel']]

# One-hot encoding for categorical variables

General_Encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_array = General_Encoder.fit_transform(X_Categorical) #this is the encoded array

encoded_cols = General_Encoder.get_feature_names_out() #back to dataframe
X_cat_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=X.index)


X_full = pd.concat([X_numeric, X_cat_df], axis=1)


sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_full, y)


# Step 3: Split into train/val sets
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)
print("Confusion matrix: ",confusion_matrix(y_val, y_pred))
print("classification Report: ",classification_report(y_val, y_pred))
