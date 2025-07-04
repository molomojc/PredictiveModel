from getData import DataFrame
import pandas as pd

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
          
    def print_head(self):
        print(self.data.head())
          
    def extract_columns(self, type="train"):
        print("Extract Features...")
        if type == 'train':
            data = {
                "DTLeadCreated": self.data['DTLeadCreated'],
                "DTLeadAllocated": self.data['DTLeadAllocated'],         
                "OBSFullName": self.data['OBSFullName'],
                "OBSEmail": self.data['OBSEmail'],         
                "CellPhoneNoLength": self.data['CellPhoneNoLength'],
                "HourOfEnquiry": self.data['HourOfEnquiry'],
                "DayOfEnquiry": self.data['DayOfEnquiry'],
                "InFinanceProcessSystemApp": self.data['InFinanceProcessSystemApp'],
                "FinanceApplied": self.data['FinanceApplied'],
                "FinanceApproved": self.data['FinanceApproved'],
                "VehicleSold": self.data['VehicleSold']
            }
        elif type == 'test':
            data = {
                "DTLeadCreated": self.data['DTLeadCreated'],
                "DTLeadAllocated": self.data['DTLeadAllocated'],         
                "OBSFullName": self.data['OBSFullName'],
                "OBSEmail": self.data['OBSEmail'],         
                "CellPhoneNoLength": self.data['CellPhoneNoLength'],
                "HourOfEnquiry": self.data['HourOfEnquiry'],
                "DayOfEnquiry": self.data['DayOfEnquiry'],
            }
        else:
            raise ValueError("Type must be either 'train' or 'test'")
          
        return pd.DataFrame(data)
     
    def Extract_Category(self):
        print("Extracting category features..")  
             
        Dealer = self.data['Dealer']
        LeadSource = self.data['LeadSource']
        LeadType = self.data['LeadType']
        Seek = self.data['Seek']
        InterestMake = self.data['InterestMake']
        InterestModel = self.data['InterestModel']
        Domain = self.data['Domain']  
           
        self.Validate_Prefix()    
        CellPrefix = self.data['CellPrefix']    
         
         
        print(f"CellPrefix: {CellPrefix}")
         
        data = {
            "Dealer_Encoder": pd.get_dummies(Dealer, prefix='Dealer', drop_first=True), 
            "LeadSource_Encoder": pd.get_dummies(LeadSource, prefix='LeadSource', drop_first=True),
            "LeadType_Encoder": pd.get_dummies(LeadType, prefix='LeadType', drop_first=True),
            "Seek_Encoder": pd.get_dummies(Seek, prefix='Seek', drop_first=True),
            "InterestMake_Encoder": pd.get_dummies(InterestMake, prefix='InterestMake', drop_first=True),
            "InterestModel_Encoder": pd.get_dummies(InterestModel, prefix='InterestModel', drop_first=True),
            "Domain_Encoder": pd.get_dummies(Domain, prefix='Domain', drop_first=True),
            "CellPrefix_Encoder": pd.get_dummies(CellPrefix, prefix='CellPrefix', drop_first=True),
        }
         
        encoded_df = pd.concat(data.values(), axis=1)
       
        return encoded_df
     
    def Validate_Prefix(self):
        print("Validating Cell Prefix...")
        pattern = r'^(?:[1-5][0-9]|[6-8][0-9])$'  
        
        self.data['CellPrefix'] = self.data['CellPrefix'].astype(str)
        
        mask = self.data['CellPrefix'].str.match(pattern)
        
        # Keep only matching prefixes, else set to NaN or a default invalid value
        self.data.loc[mask, 'CellPrefix'] = pd.NA
        
        # Convert valid prefixes to integer type (skip NaNs)
        self.data.loc[mask, 'CellPrefix'] = self.data.loc[mask, 'CellPrefix'].astype(int)

            
            
           
Model = FeatureExtractor('train')
Model.Extract_Category()


