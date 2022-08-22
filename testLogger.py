import numpy as np
import tensorflow as tf
from models.logger import FullLogger

def test1():
    fullLogger = FullLogger()
    fullLogger.SetNewEpochs()
    A = tf.zeros([2, 3, 4], tf.int32)
    B = tf.zeros([4, 3], tf.int32)
    C = tf.matmul(A, B)
    fullLogger.AddNewLog([A.shape, B.shape], "matmul")
    fullLogger.AddNewLog([A.shape], "transpose")
    fullLogger.AddNewLog([A.shape, B.shape], "matmul")
    fullLogger.SetNewEpochs()
    fullLogger.AddNewLog([A.shape, B.shape], "matmul")
    fullLogger.AddNewLog([A.shape, B.shape], "matmul")
    fullLogger.ShowLog()
    fullLogger.WriteLog('operationLog.txt')
    fullLogger.ShowNaiveResult(0)
    fullLogger.ShowImproveResult(0)
    fullLogger.ShowNaiveResult(1)
    fullLogger.ShowImproveResult(1)

def test2():
    fullLogger = FullLogger()
    fullLogger.ReadLog('operationLog.txt')
    fullLogger.ShowLog()

def test():
    #test1()
    test2()

if __name__ == "__main__":
    test()