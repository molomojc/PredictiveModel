import pandas as pd
from Data import Dataframe
from sklearn.impute import SimpleImputer   

class DataCleanup:
    def __init__(self,type):
        self.data = Dataframe(type)
        
    #Main function to clean the data
    def cleanData(self, type="train"):
        
        data = self.data.getDataFrame()
          
        if type == "train":
            data = data.drop(columns=[
        'InFinanceProcessSystemApp', 'FinanceApplied', 'FinanceApproved', 
        'OBSFullName', 'OBSEmail'
      ])     
            
        elif type == "test":
            data = data.drop(columns=['OBSFullName', 'OBSEmail'])
            
        
        data['DTLeadCreated'] = pd.to_datetime(data['DTLeadCreated'], errors='coerce').astype('int64') / 10**9
        data['DTLeadAllocated'] = pd.to_datetime(data['DTLeadAllocated'], errors='coerce').astype('int64') / 10**9
        
        #replace Nan values
        Imputer = SimpleImputer(strategy='most_frequent')
        data = pd.DataFrame(Imputer.fit_transform(data), columns=data.columns)
       
        data['Domain'] = data['Domain'].str.extract(r'@([a-zA-Z0-9.-]+\.(?:com|co\.za|ac\.za|gov\.za))', expand=False)
       #Fix cell prefix for invalid values
        data['CellPrefix'] = self.validate_cell_prefix(self.data)
        
        data['Dealer'] = self.reduce_cardinality(data['Dealer'], 30)
        data['Domain'] = self.reduce_cardinality(data['Domain'], 15)
        data['InterestMake'] = self.reduce_cardinality(data['InterestMake'], 20)
        data['InterestModel'] = self.reduce_cardinality(data['InterestModel'], 20)

        #replace Nan values
        #NOTE: we have only used personal emails we have to use the company ones after
        Imputer = SimpleImputer(strategy='most_frequent')
        data = pd.DataFrame(Imputer.fit_transform(data), columns=data.columns)
        
        '''
        list = set()
       
        for domain in data['Domain']:
            if domain not in list:
                list.add(domain)
  
        print("Unique domains:", list)
        '''
        
        
        return data  
    
    def validate_cell_prefix(self, data):
      pattern = r'(?:[1-5][0-9]|[6-8][0-9])$'
      data = self.data.getDataFrame()
      print("converting CellPrefix to string and removing non-numeric characters")
      data['CellPrefix'] = data['CellPrefix'].astype(str)
      data['CellPrefix'] = data['CellPrefix'].str.replace(r'[^0-9]', '', regex=True)
      print("Filtering CellPrefix with regex pattern:", pattern)
      matched_values = data['CellPrefix'][data['CellPrefix'].str.match(pattern)]
      print("Converting matched CellPrefix values to integers")
      matched_values_to_int = matched_values.astype(int)
      
      return matched_values_to_int
    
    #Reduce the number of unique values in cols
    def reduce_cardinality(self, col, top_n=20):
        top = col.value_counts().nlargest(top_n).index
        return col.where(col.isin(top), 'Other')


Model = DataCleanup('train')
cleaned_train = Model.cleanData('train')
print("Number of null values:", cleaned_train.isnull().sum().sum())

print(cleaned_train)


            
            
        


