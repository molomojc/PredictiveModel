import pandas as pd
from DataCleanup import DataCleanup as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

# ======== TRAINING DATA PHASE ========
data = dc("train").cleanData()  # Clean the training data

# Prepare target
y = pd.to_numeric(data['VehicleSold'], errors='coerce').fillna(0).astype(int)
lead_ids = data['LeadID'].reset_index(drop=True)
# Drop non-feature columns
X = data.drop(columns=['VehicleSold', 'CustomerID', 'LeadID'])

# Split into numeric and categorical
X_numeric = X[['CellPhoneNoLength', 'HourOfEnquiry', 'DayOfEnquiry', 'DTLeadCreated', 'DTLeadAllocated']]
X_categorical = X[['LeadType', 'LeadSource', 'CellPrefix', 'Domain', 'Dealer', 'Seek', 'InterestMake', 'InterestModel']]

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_array = encoder.fit_transform(X_categorical)
encoded_cols = encoder.get_feature_names_out()
X_cat_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=X.index)

# Combine numeric + encoded categorical
X_full = pd.concat([X_numeric, X_cat_df], axis=1)

# Handle imbalance with SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_full, y)

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification report:\n", classification_report(y_val, y_pred)), 



'''
# ======== TESTING PHASE ========
test_data = dc("test").cleanData("test")  # Clean test data
lead_ids = test_data['LeadID'].reset_index(drop=True)

# Drop non-feature columns
X_test = test_data.drop(columns=['CustomerID', 'LeadID'])

# Separate test numeric and categorical
X_test_numeric = X_test[['CellPhoneNoLength', 'HourOfEnquiry', 'DayOfEnquiry', 'DTLeadCreated', 'DTLeadAllocated']]
X_test_categorical = X_test[['LeadType', 'LeadSource', 'CellPrefix', 'Domain', 'Dealer', 'Seek', 'InterestMake', 'InterestModel']]

# Use the same encoder from training
encoded_test_array = encoder.transform(X_test_categorical)
X_test_cat_df = pd.DataFrame(encoded_test_array, columns=encoded_cols, index=X_test.index)

# Combine numeric + encoded
X_test_full = pd.concat([X_test_numeric, X_test_cat_df], axis=1)

# Predict probabilities
#
probs = model.predict_proba(X_test_full)[:, 1]

# Output results
results = pd.DataFrame({
    'LeadID': lead_ids,
    'VehicleSoldProbability': probs
})

# Sort by highest probability
results_sorted = results.sort_values(by='VehicleSoldProbability', ascending=False)

# Show top 10 leads most likely to convert
print("\nTop 10 predicted leads with highest probability:")
print(results_sorted.head(10))

# Optionally save
results_sorted.to_csv("lead_sale_probabilities.csv", index=False, float_format='%.10f')
'''