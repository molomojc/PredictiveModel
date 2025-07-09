import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from DataCleanup import DataCleanup as dc
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

#TRAINING DATA PHASE
data = dc("train").cleanData() #called the DataCleanup class to clean the data
y = pd.to_numeric(data['VehicleSold'], errors='coerce')  # Convert to numeric, force bad values to NaN
y = y.fillna(0).astype(int)  # Replace NaNs and convert to int
X = data.drop(columns=['VehicleSold' , 'CustomerID', 'LeadID']) #features excluding the target variable 
# One-hot encoding for categorical variables

X_Categorical = X[['LeadType', 'LeadSource', 'CellPrefix', 'Domain', 'Dealer','Seek', 'InterestMake', 'InterestModel']]

General_Encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = General_Encoder.fit_transform(X_Categorical) #this is the encoded array

print("Encoded array shape:", X_encoded.shape)

encoded_cols = General_Encoder.get_feature_names_out()
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_cols, index=X.index)

model = LogisticRegression(max_iter=500, class_weight='balanced')
model.fit(X_encoded_df, y)

data_test = dc("test").cleanData("test") #called the DataCleanup class to clean the data
X_test = data_test.drop(columns=['CustomerID', 'LeadID']) #features excluding the target variable

# Extract categorical columns from test set, same as train
X_test_categorical = X_test[['LeadType', 'LeadSource', 'CellPrefix', 'Domain', 'Dealer', 'Seek', 'InterestMake', 'InterestModel']]

# Use the SAME encoder fitted on training data to transform test data
encoded_test_array = General_Encoder.transform(X_test_categorical)

# Train-Test split 80/20
X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y, test_size = 0.2, random_state= 42, stratify=y) 

#Using SMOTE to oversample minority class, since recall, precision and f1-score are 0.00
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

#Train XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

#predict with 20%
y_Pred=model.predict(X_test)

print("Test Accuracy:",accuracy_score(y_test, y_Pred))
print("Confusion matrix:\n", confusion_matrix(y_test,  y_Pred))
print("Train Label Distribution:\n", y.value_counts(normalize=True))

print(classification_report(y_test,y_Pred))

'''
# Create DataFrame for test encoded features with same columns as training
X_test_cat_df = pd.DataFrame(encoded_test_array, columns=encoded_cols, index=X_test.index)

# Make predictions on test data
test_predictions = model.predict(X_test_cat_df)

print("Test predictions:", test_predictions[:500])
print("From the Train set:",y.value_counts(normalize=True))
print("From the Test set:", pd.Series(test_predictions))
'''