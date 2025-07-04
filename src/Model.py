from getData import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class FeatureExtractor:
    
      p = DataFrame('train')
     
      
      def __init__(self):
          self.data = self.p.getDataFrame().head(10)
          
      def print_head(self):
            print(self.data.head())
          
        # This method extracts features from the data.
      def extract_columns(self):
          print("Extract Feateures...")
          Customer_ID = self.data['CustomerID']
          LeadID = self.data['LeadID']
          DTLeadCreated = self.data['DTLeadCreated']
          DTLeadAllocated = self.data['DTLeadAllocated']         
          OBSFullName = self.data['OBSFullName']
          OBSEmail = self.data['OBSEmail']         
          CellPhoneNoLength = self.data['CellPhoneNoLength']
          HourOfEnquiry = self.data['HourOfEnquiry']
          DayOfEnquiry = self.data['DayOfEnquiry']
          InFinanceProcessSystemApp = self.data['InFinanceProcessSystemApp']
          FinanceApplied = self.data['FinanceApplied']
          FinanceApproved = self.data['FinanceApproved']
          VehicleSold = self.data['VehicleSold']
     
      def Extract_Category(self):
         print("EXtracting categpory features..")     
         Dealer = self.data['Dealer']
         LeadSource = self.data['LeadSource']
         LeadType = self.data['LeadType']
         Seek = self.data['Seek']
         InterestMake = self.data['InterestMake']
         InterestModel = self.data['InterestModel']
         Domain = self.data['Domain']        
         CellPrefix = self.data['CellPrefix']
         
         Dealer_Encoder = pd.get_dummies(Dealer, prefix='Dealer', drop_first=True)
         LeadSource_Encoder = pd.get_dummies(LeadSource, prefix='LeadSource', drop_first=True)
         LeadType_Encoder = pd.get_dummies(LeadType, prefix='LeadType', drop_first=True)
         Seek_Encoder = pd.get_dummies(Seek, prefix='Seek', drop_first=True)
         InterestMake_Encoder = pd.get_dummies(InterestMake, prefix='InterestMake', drop_first=True)
         InterestModel_Encoder = pd.get_dummies(InterestModel, prefix='InterestModel', drop_first=True)
         Domain_Encoder = pd.get_dummies(Domain, prefix='Domain', drop_first=True)
         CellPrefix_Encoder = pd.get_dummies(CellPrefix, prefix='CellPrefix', drop_first=True)
         Dealer_Encoder = Dealer_Encoder.astype(int)
    
         print(Dealer_Encoder.to_string())
         
       #   print(f"Extracted columns: {Customer_ID}, {LeadID}, {DTLeadCreated}, {DTLeadAllocated}, {Dealer}, {LeadSource}, {LeadType}, {Seek}, {InterestMake}, {InterestModel}, {OBSFullName}, {OBSEmail}, {Domain}, {CellPrefix}, {CellPhoneNoLength}, {HourOfEnquiry}, {DayOfEnquiry}, {InFinanceProcessSystemApp}, {FinanceApplied}, {FinanceApproved}, {VehicleSold}")            
           
           
           
Model = FeatureExtractor()
#Model.print_head()
Model.Extract_Category() 


