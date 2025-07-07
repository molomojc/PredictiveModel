from getData import DataFrame
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class FeatureExtractor:  
    def __init__(self, type):
        if type == 'train':
            print("Loading training data...")
            self.data = DataFrame('train').getDataFrame()
        elif type == 'test':
            print("Loading testing data...")
            self.data = DataFrame('test').getDataFrame()
        else:
            raise ValueError("Type must be either 'train' or 'test'")
        
        # Track which mode we're in
        self.type = type
        
        # Save column names
        self.cat_cols = ['Dealer', 'LeadSource', 'LeadType', 'Seek', 
                         'InterestMake', 'InterestModel', 'Domain', 'CellPrefix']
        self.num_cols = ['CellPhoneNoLength', 'HourOfEnquiry', 'DayOfEnquiry']
        self.encoder = None  # Will hold OneHotEncoder

    def print_head(self):
        print(self.data.head())

    def extract_columns(self):
        print("Extracting numeric & target features...")
        
        df = self.data.copy()

        if self.type == 'train':
            features = df[self.num_cols + [
                'DTLeadCreated', 'DTLeadAllocated', 'OBSFullName', 'OBSEmail',
                'InFinanceProcessSystemApp', 'FinanceApplied', 'FinanceApproved', 'VehicleSold'
            ]]
        else:
            features = df[self.num_cols + [
                'DTLeadCreated', 'DTLeadAllocated', 'OBSFullName', 'OBSEmail'
            ]]

        return features

    def Extract_Category(self, encoder=None):
        print("Extracting categorical features with OneHotEncoder...")

        df = self.data.copy()
        df['CellPrefix'] = self.validate_cell_prefix(df)

        # Subset for relevant categorical columns
        df_cat = df[self.cat_cols].copy()

        # Fill missing values first (or do it via SimpleImputer in pipeline if needed)
        df_cat = df_cat.fillna('Missing')

        if self.type == 'train':
            # Fit encoder on training data
            self.encoder = OneHotEncoder(handle_unknown='ignore')
            encoded_array = self.encoder.fit_transform(df_cat)
        else:
            # Use provided encoder (from train)
            if encoder is None:
                raise ValueError("You must provide a fitted encoder for test data.")
            self.encoder = encoder
            encoded_array = self.encoder.transform(df_cat)

        encoded_df = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out(self.cat_cols))
        encoded_df.index = df_cat.index  # Keep index alignment

        return encoded_df

    def validate_cell_prefix(self, data):
      pattern = r'(?:[1-5][0-9]|[6-8][0-9])$'
      data = data.copy()  

      data['CellPrefix'] = data['CellPrefix'].astype(str)
      data['CellPrefix'] = data['CellPrefix'].str.replace(r'[^0-9]', '', regex=True)

      matched_values = data['CellPrefix'][data['CellPrefix'].str.match(pattern)]
      matched_values_to_int = matched_values.astype(int)
      return matched_values_to_int

