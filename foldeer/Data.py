import pandas as pd

class Dataframe:
    
    def __init__(self, type):
        if type == 'train':
            print("Loading training data...")
            self.data =  pd.read_csv('https://www.mxhackathon.co.za/docs/TrainData.csv')
        elif type == 'test':
            print("Loading test data...")
            self.data = pd.read_csv('https://www.mxhackathon.co.za/docs/TestData.csv')
            
    def getDataFrame(self):
        return self.data
    
