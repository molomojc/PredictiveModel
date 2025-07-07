import pandas as pd


class DataFrame:
    
    Train_Data = pd.read_csv('data/TrainData.csv')
    Test_Data = pd.read_csv('data/TestData.csv')
    
    # This class is used to get the DataFrame for either training or testing data.
    def __init__(self,type):
        if type == 'train':
            print("Loading training data...")
            self.data = self.Train_Data
        elif type == 'test':
            print("Loading testing data...")
            self.data = self.Test_Data
        else:
            raise ValueError("Type must be either 'train' or 'test'")
    
    #returns the actual dataframe 
    def getDataFrame(self):
      self.data = self.data.dropna()
      return self.data
    