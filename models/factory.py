from .counter import Counter, MatmulCounter, TransposeCounter

class CounterFactory:
    def __init__(self):
        self.matmulNums = 0
        self.transposeNums = 0

    def GetCounter(self, _type):
        if _type == "matmul":
            newCounter = MatmulCounter(self.matmulNums)
            self.matmulNums += 1
            return newCounter
        elif _type == "transpose":
            newCounter = TransposeCounter(self.transposeNums)
            self.transposeNums += 1
            return newCounter

    def GetCounterByMetadata(self, _metaData):
        counterType = _metaData[1]
        counterId = int(_metaData[2])
        if counterType == "matmul":
            newCounter = MatmulCounter(counterId)
            tensors = [eval(_metaData[3]), eval(_metaData[4])]
            newCounter.SetLog(tensors)
        elif counterType == "transpose":
            newCounter = TransposeCounter(counterId)
            tensors = [eval(_metaData[3])]
            newCounter.SetLog(tensors)
        return newCounter
