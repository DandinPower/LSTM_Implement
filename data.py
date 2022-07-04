import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class Dataset():
    def __init__(self,split_rate,batch_size):
        usecols = ['Open','Close','High','Low','Volume','NumOfTrade']
        df = pd.read_csv('klines.csv', usecols=usecols)
        y = df['Close']
        scaler=MinMaxScaler(feature_range=(0,1))
        y=scaler.fit_transform(y.to_frame())
        self.y = np.reshape(y, newshape=(365))
        scaler1=MinMaxScaler(feature_range=(0,1))
        self.x=scaler1.fit_transform(df)
        self.train_length = int(len(self.x) * split_rate)
        self.batch_size = batch_size

    def GetTrainData(self):
        x_original = self.x[:self.train_length]
        total = len(x_original) - self.batch_size
        for i in range(total):
            if i == 0:
                x_data = [x_original[i:i+self.batch_size]]
            else:
                x_data = tf.concat((x_data,[x_original[i:i+self.batch_size]]),0)
        y_data = self.y[self.batch_size:self.train_length]
        y_data = np.reshape(y_data, (total,1))
        return x_data, y_data
    
    def GetTestData(self):
        return self.x[self.train_length:], self.y[self.train_length:]
    
if __name__ == "__main__":
    dataset = Dataset(0.9,20)
    print(dataset.GetTrainData()[0].shape)
    print(dataset.GetTrainData()[1].shape)
    #print(dataset.GetTestData()[0].shape)