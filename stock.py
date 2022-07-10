import tensorflow as tf 
import numpy as np 
from data import Dataset
from lstm import LSTM, LinearLayer, QuantLSTM, QuantLinearLayer
from train import Train
from skrm import SKRM
SPLIT_RATE = 0.9
INPUT_DIM = 6 
HIDDEN_1 = 10
HIDDEN_2 = 5

class Stock(tf.keras.Model):
    def __init__(self, input_dim, hidden_1, hidden_2):
        super(Stock, self).__init__()
        self.lstm = LSTM(input_dim, hidden_1)
        self.dense1 = LinearLayer(hidden_1, hidden_2)
        self.dense2 = LinearLayer(hidden_2, 1)

    def call(self, inputs):
        output1 = self.lstm(inputs)
        output2 = self.dense1(output1)
        output3 = self.dense2(output2)
        return output3 

class QuantStock(tf.keras.Model):
    def __init__(self, input_dim, hidden_1, hidden_2):
        super(QuantStock, self).__init__()
        self.lstm = QuantLSTM(input_dim, hidden_1)
        self.dense1 = QuantLinearLayer(hidden_1, hidden_2)
        self.dense2 = QuantLinearLayer(hidden_2, 1)

    def call(self, inputs):
        output1 = self.lstm(inputs)
        output2 = self.dense1(output1)
        output3 = self.dense2(output2)
        return output3 

def Normal():
    dataset = Dataset(SPLIT_RATE,10)
    train_data_x, train_data_y = dataset.GetTrainData()
    model = Stock(INPUT_DIM, HIDDEN_1, HIDDEN_2)
    Train(model, train_data_x, train_data_y, 50)

def Quant():
    dataset = Dataset(SPLIT_RATE,10)
    train_data_x, train_data_y = dataset.GetTrainData()
    model = QuantStock(INPUT_DIM, HIDDEN_1, HIDDEN_2)
    Train(model, train_data_x, train_data_y, 50)

def SKRM():
    skrms = SKRM()
    dataset = Dataset(SPLIT_RATE,10)
    train_data_x, train_data_y = dataset.GetTrainData()
    model = QuantStock(INPUT_DIM, HIDDEN_1, HIDDEN_2)
    Train(model, train_data_x, train_data_y, 50)


if __name__ == "__main__":
    Normal()
    Quant()
    
    