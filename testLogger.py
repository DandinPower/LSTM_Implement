import numpy as np
import tensorflow as tf
from models.logger import FullLogger
from models.counter import MatmulCounter, TransposeCounter

#測試寫入幾個operation並驗證正確性
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

#測試可不可以正常寫入txt檔
def test2():
    fullLogger = FullLogger()
    fullLogger.ReadLog('operation.txt')
    fullLogger.ShowLog()

#測試可不可以正確計算出float的數量
def test3():
    matmul = MatmulCounter(0)
    matmul.SetLog([[2, 3, 4], [4, 5]])
    print(matmul.floatNumsInTensors)

#測試TransposeCounter計算是否正確
def test4():
    transpose = TransposeCounter(0)
    transpose.SetLog([[3, 4, 5]])
    print(transpose.GetSkrmNaiveRecord())

#測試MatmulCounter計算IJK是否正確
def test5():
    matmul = MatmulCounter(0)
    matmul.SetBlockSize([[2, 2], [2, 2]])
    matmul.SetLog([[1,20], [20, 5]])
    matmul.ShowIJK()

#測試MatmulCounter計算Naive的情況
def test6():
    matmul = MatmulCounter(0)
    matmul.SetBlockSize([[2, 2], [2, 2]])
    matmul.SetLog([[1,20], [20, 5]])
    print(matmul.GetSkrmNaiveRecord())

#測試MatmulCounter計算Improve的情況
def test7():
    matmul = MatmulCounter(0)
    matmul.SetBlockSize([[2, 2], [2, 2]])
    matmul.SetLog([[1,20], [20, 5]])
    print(matmul.GetSkrmImproveRecord())

def test():
    #test1()
    #test2()
    #test3()
    #test4()
    #test5()
    test6()
    test7()
    
if __name__ == "__main__":
    test()