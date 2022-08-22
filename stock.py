import tensorflow as tf 
import numpy as np 
from data import Dataset
from lstm import LSTM, LinearLayer, QuantLSTM, QuantLinearLayer, SKRMLSTM, ErrorLSTM, ErrorLinearLayer, OperationLSTM
from train import Train, WriteHistory, Train_V2
from models.logger import FullLogger
from skrm import SKRM
from dotenv import load_dotenv
import os
load_dotenv()

ERROR_RATE = float(os.getenv('ERROR_RATE'))
FLIP_START = int(os.getenv('FLIP_START'))
FLIP_END = int(os.getenv('FLIP_END'))
EPOCHS = int(os.getenv('EPOCHS'))

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

class OperationStock(tf.keras.Model):
    def __init__(self, logger, input_dim, hidden_1, hidden_2):
        super(OperationStock, self).__init__()
        self.logger = logger
        self.lstm = OperationLSTM(logger, input_dim, hidden_1)
        self.dense1 = LinearLayer(hidden_1, hidden_2)
        self.dense2 = LinearLayer(hidden_2, 1)

    def call(self, inputs):
        output1 = self.lstm(inputs)
        output2 = self.dense1(output1)
        output3 = self.dense2(output2)
        self.logger.AddNewLog([output1.shape, self.dense1.w], "matmul")
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

class SKRMStock(tf.keras.Model):
    def __init__(self, input_dim, hidden_1, hidden_2, skrms):
        super(SKRMStock, self).__init__()
        self.skrms = skrms
        self.lstm = SKRMLSTM(input_dim, hidden_1, skrms)
        self.dense1 = LinearLayer(hidden_1, hidden_2)
        self.dense2 = LinearLayer(hidden_2, 1)

    def call(self, inputs):
        output1 = self.lstm(inputs)
        output2 = self.dense1(output1)
        output3 = self.dense2(output2)
        self.skrms.Count(output1, output2)
        self.skrms.Count(output2, output3)
        return output3 

class ErrorStock(tf.keras.Model):
    def __init__(self, input_dim, hidden_1, hidden_2):
        super(ErrorStock, self).__init__()
        self.lstm = ErrorLSTM(input_dim, hidden_1)
        self.dense1 = ErrorLinearLayer(hidden_1, hidden_2)
        self.dense2 = ErrorLinearLayer(hidden_2, 1)

    def call(self, inputs):
        output1 = self.lstm(inputs)
        output2 = self.dense1(output1)
        output3 = self.dense2(output2)
        return output3 

def Normal():
    dataset = Dataset(SPLIT_RATE,10)
    train_data_x, train_data_y = dataset.GetTrainData()
    model = Stock(INPUT_DIM, HIDDEN_1, HIDDEN_2)
    history = Train(model, train_data_x, train_data_y, EPOCHS)
    WriteHistory(history, 'history/original.txt')

def Quant():
    dataset = Dataset(SPLIT_RATE,10)
    train_data_x, train_data_y = dataset.GetTrainData()
    model2 = QuantStock(INPUT_DIM, HIDDEN_1, HIDDEN_2)
    history = Train(model2, train_data_x, train_data_y, EPOCHS)
    WriteHistory(history, 'history/approximate.txt')

def Skrm():
    skrms = SKRM()
    dataset = Dataset(SPLIT_RATE,10)
    train_data_x, train_data_y = dataset.GetTrainData()
    model3 = SKRMStock(INPUT_DIM, HIDDEN_1, HIDDEN_2, skrms)
    Train(model3, train_data_x, train_data_y, EPOCHS)
    print(skrms.GetCount())

def Error():
    dataset = Dataset(SPLIT_RATE,10)
    train_data_x, train_data_y = dataset.GetTrainData()
    model4 = ErrorStock(INPUT_DIM, HIDDEN_1, HIDDEN_2)
    history = Train(model4, train_data_x, train_data_y, EPOCHS)
    WriteHistory(history, 'history/22.txt')

def Operation():
    fullLogger = FullLogger()
    dataset = Dataset(SPLIT_RATE,10)
    train_data_x, train_data_y = dataset.GetTrainData()
    model = OperationStock(fullLogger, INPUT_DIM, HIDDEN_1, HIDDEN_2)
    history = Train_V2(fullLogger, model, train_data_x, train_data_y, EPOCHS)
    fullLogger.WriteLog('operation.txt')

if __name__ == "__main__":
    #Normal()
    #Quant()
    #Skrm()
    #Error()
    Operation()
    
    