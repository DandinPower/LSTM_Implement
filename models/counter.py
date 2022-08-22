import numpy as np 
import tensorflow as tf 

class Counter:
    def __init__(self, *args):
        self.id = args[0]
        self.tensors = []

    def SetLog(self, _tensors):
        self.tensors.clear()
        for tensor in _tensors:
            temp = []
            for dim in tensor:
                temp.append(dim)
            self.tensors.append(temp)

    def ShowLog(self):
        pass

    def GetSkrmNaiveRecord(self):
        pass

    def GetSkrmImproveRecord(self):
        pass

class MatmulCounter(Counter):
    def __init__(self, *args):
        super().__init__(*args)

    def ShowLog(self):
        return f'matmul;{self.id};{self.tensors[0]};{self.tensors[1]}'

    def GetSkrmNaiveRecord(self):
        return [0, 0, 0, 0]

    def GetSkrmImproveRecord(self):
        return [0, 0, 0, 0]

class TransposeCounter(Counter):
    def __init__(self, *args):
        super().__init__(*args)

    def ShowLog(self):
        return f'transpose;{self.id};{self.tensors[0]}'

    def GetSkrmNaiveRecord(self):
        return [0, 0, 0, 0]

    def GetSkrmImproveRecord(self):
        return [0, 0, 0, 0]