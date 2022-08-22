import tensorflow as tf 
import numpy as np 
from tensorflow.python.framework import ops
from skrm import SKRM
from dotenv import load_dotenv
import os
load_dotenv()

ERROR_RATE = float(os.getenv('ERROR_RATE'))
FLIP_START = int(os.getenv('FLIP_START'))
FLIP_END = int(os.getenv('FLIP_END'))

@ops.RegisterGradient("BitsQuant")
def _bits_quant_grad(op, grad):
    return [grad] 

@ops.RegisterGradient("CountSkrm")
def _count_skrm_grad(op, grad):
  return [grad] 

@ops.RegisterGradient("RandomError")
def _bits_quant_grad(op, grad):
  inputs = op.inputs[0]
  return [grad] 

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.w = self.add_variable(name='w',
            shape=[input_dim, output_dim], initializer="random_normal", trainable = True)
        self.b = self.add_variable(name='b',
            shape=[output_dim], initializer="random_normal", trainable = True)

    def call(self, inputs):
        matmul = tf.matmul(inputs, self.w)
        bias = matmul + self.b
        return bias

class QuantLinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.kernel = tf.load_op_library('./bits_quant.so')
        self.w = self.add_variable(name='w',
            shape=[input_dim, output_dim], initializer="random_normal", trainable = True)
        self.b = self.add_variable(name='b',
            shape=[output_dim], initializer="random_normal", trainable = True)

    def call(self, inputs):
        y_pred = self.kernel.bits_quant(tf.matmul(inputs, self.w)) + self.kernel.bits_quant(self.b)
        return self.kernel.bits_quant(y_pred)

class LSTMCell(tf.keras.Model):
    def __init__(self, input_dim):
        super(LSTMCell, self).__init__()
        self.c = self.add_variable(name='memory_cell', shape=[1, input_dim], initializer="random_normal", trainable = True)
        self.h = self.add_variable(name='last_output', shape=[1, input_dim], initializer="random_normal", trainable = True)
        self.Wf = LinearLayer(input_dim * 2, input_dim)
        self.Wi = LinearLayer(input_dim * 2, input_dim)
        self.Wc = LinearLayer(input_dim * 2, input_dim)
        self.Wo = LinearLayer(input_dim * 2, input_dim)
        self.dims = input_dim

    def call(self, inputs):
        concat = tf.concat([self.h, inputs],1)
        z = tf.keras.activations.tanh(self.Wc(concat))
        zf = tf.keras.activations.sigmoid(self.Wf(concat))
        zi = tf.keras.activations.sigmoid(self.Wi(concat))
        zo = tf.keras.activations.sigmoid(self.Wo(concat))
        memory = z * zi
        self.c = self.c * zf + memory
        self.h = zo * tf.keras.activations.tanh(self.c)
        return self.h

class OperationCell(tf.keras.Model):
    def __init__(self, logger, input_dim):
        super(OperationCell, self).__init__()
        self.logger = logger
        self.c = self.add_variable(name='memory_cell', shape=[1, input_dim], initializer="random_normal", trainable = True)
        self.h = self.add_variable(name='last_output', shape=[1, input_dim], initializer="random_normal", trainable = True)
        self.Wf = LinearLayer(input_dim * 2, input_dim)
        self.Wi = LinearLayer(input_dim * 2, input_dim)
        self.Wc = LinearLayer(input_dim * 2, input_dim)
        self.Wo = LinearLayer(input_dim * 2, input_dim)
        self.dims = input_dim

    def call(self, inputs):
        concat = tf.concat([self.h, inputs],1)
        z = tf.keras.activations.tanh(self.Wc(concat))
        zf = tf.keras.activations.sigmoid(self.Wf(concat))
        zi = tf.keras.activations.sigmoid(self.Wi(concat))
        zo = tf.keras.activations.sigmoid(self.Wo(concat))
        memory = z * zi
        self.c = self.c * zf + memory
        self.h = zo * tf.keras.activations.tanh(self.c)
        return self.h

class OperationLSTM(tf.keras.Model):
    def __init__(self, logger, input_dim, hidden_dim):
        super(OperationLSTM, self).__init__()
        self.logger = logger
        self.linear = LinearLayer(input_dim, hidden_dim)
        self.cell = OperationCell(logger, hidden_dim)

    def call(self, inputs):
        inputs = self.linear(inputs)
        for i in range(len(inputs)):
            for data in inputs[i]:
                output = self.cell([data])
            if i == 0:
                outputs = output
            else:
                outputs = tf.concat((outputs, output),0)
        return outputs

class QuantLSTMCell(tf.keras.Model):
    def __init__(self, input_dim):
        super(QuantLSTMCell, self).__init__()
        self.kernel = tf.load_op_library('./bits_quant.so')
        self.c = self.add_variable(name='memory_cell', shape=[1, input_dim], initializer="random_normal", trainable = True)
        self.h = self.add_variable(name='last_output', shape=[1, input_dim], initializer="random_normal", trainable = True)
        self.Wf = QuantLinearLayer(input_dim * 2, input_dim)
        self.Wi = QuantLinearLayer(input_dim * 2, input_dim)
        self.Wc = QuantLinearLayer(input_dim * 2, input_dim)
        self.Wo = QuantLinearLayer(input_dim * 2, input_dim)
        self.dims = input_dim

    def call(self, inputs):
        concat = self.kernel.bits_quant(tf.concat([self.h, inputs],1))
        z = self.kernel.bits_quant(tf.keras.activations.tanh(self.Wc(concat)))
        zf = self.kernel.bits_quant(tf.keras.activations.sigmoid(self.Wf(concat)))
        zi = self.kernel.bits_quant(tf.keras.activations.sigmoid(self.Wi(concat)))
        zo = self.kernel.bits_quant(tf.keras.activations.sigmoid(self.Wo(concat)))
        memory = self.kernel.bits_quant(z * zi)
        self.c = self.kernel.bits_quant(self.kernel.bits_quant(self.c * zf) + memory)
        self.h = self.kernel.bits_quant(zo * self.kernel.bits_quant(tf.keras.activations.tanh(self.c)))
        return self.h

class SKRMLSTMCell(tf.keras.Model):
    def __init__(self, input_dim, skrms):
        super(SKRMLSTMCell, self).__init__()
        self.skrms = skrms
        self.c = self.add_variable(name='memory_cell', shape=[1, input_dim], initializer="random_normal", trainable = True)
        self.h = self.add_variable(name='last_output', shape=[1, input_dim], initializer="random_normal", trainable = True)
        self.Wf = LinearLayer(input_dim * 2, input_dim)
        self.Wi = LinearLayer(input_dim * 2, input_dim)
        self.Wc = LinearLayer(input_dim * 2, input_dim)
        self.Wo = LinearLayer(input_dim * 2, input_dim)
        self.dims = input_dim

    def call(self, inputs):
        concat = tf.concat([self.h, inputs],1)
        z1 = self.Wc(concat)
        z = tf.keras.activations.tanh(z1)
        zf1 = self.Wf(concat)
        zf = tf.keras.activations.sigmoid(zf1)
        zi1 = self.Wi(concat)
        zi = tf.keras.activations.sigmoid(zi1)
        zo1 = self.Wo(concat)
        zo = tf.keras.activations.sigmoid(zo1)
        memory = z * zi
        current_c = self.c
        last = self.c * zf
        self.c =  last + memory
        tanhc = tf.keras.activations.tanh(self.c)
        self.h = zo * tanhc
        self.skrms.Count(inputs, concat)
        self.skrms.Count(concat, z1)
        self.skrms.Count(concat, zf1)
        self.skrms.Count(concat, zi1)
        self.skrms.Count(concat, zo1)
        self.skrms.Count(z1, z)
        self.skrms.Count(zf1, zf)
        self.skrms.Count(zi1, zi)
        self.skrms.Count(zo1, zo)
        self.skrms.Count(z, memory)
        self.skrms.Count(current_c, last)
        self.skrms.Count(last, self.c)
        self.skrms.Count(self.c, tanhc)
        self.skrms.Count(zo, self.h)
        return self.h

class LSTM(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.linear = LinearLayer(input_dim, hidden_dim)
        self.cell = LSTMCell(hidden_dim)

    def call(self, inputs):
        inputs = self.linear(inputs)
        for i in range(len(inputs)):
            for data in inputs[i]:
                output = self.cell([data])
            if i == 0:
                outputs = output
            else:
                outputs = tf.concat((outputs, output),0)
        return outputs

class QuantLSTM(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim):
        super(QuantLSTM, self).__init__()
        self.linear = QuantLinearLayer(input_dim, hidden_dim)
        self.cell = QuantLSTMCell(hidden_dim)

    def call(self, inputs):
        inputs = self.linear(inputs)
        for i in range(len(inputs)):
            for data in inputs[i]:
                output = self.cell([data])
            if i == 0:
                outputs = output
            else:
                outputs = tf.concat((outputs, output),0)
        return outputs

class SKRMLSTM(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, skrms):
        super(SKRMLSTM, self).__init__()
        self.skrms = skrms
        self.linear = LinearLayer(input_dim, hidden_dim)
        self.cell = SKRMLSTMCell(hidden_dim, skrms)

    def call(self, inputs):
        dense_inputs = self.linear(inputs)
        self.skrms.Count(inputs, dense_inputs)
        for i in range(len(dense_inputs)):
            for data in dense_inputs[i]:
                output = self.cell([data])
            if i == 0:
                outputs = output
                current_outputs = outputs
            else:
                current_outputs = outputs
                outputs = tf.concat((outputs, output),0)
            self.skrms.Count(current_outputs, outputs)
        return outputs

class ErrorLinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.kernel = tf.load_op_library('./random_error.so')
        self.w = self.add_variable(name='w',
            shape=[input_dim, output_dim], initializer="random_normal", trainable = True)
        self.b = self.add_variable(name='b',
            shape=[output_dim], initializer="random_normal", trainable = True)

    def call(self, inputs):
        y_pred = self.kernel.random_error(tf.matmul(inputs, self.w), ERROR_RATE, FLIP_START, FLIP_END) + self.kernel.random_error(self.b, ERROR_RATE, FLIP_START, FLIP_END)
        return self.kernel.random_error(y_pred, ERROR_RATE, FLIP_START, FLIP_END)

class ErrorLSTMCell(tf.keras.Model):
    def __init__(self, input_dim):
        super(ErrorLSTMCell, self).__init__()
        self.kernel = tf.load_op_library('./random_error.so')
        self.c = self.add_variable(name='memory_cell', shape=[1, input_dim], initializer="random_normal", trainable = True)
        self.h = self.add_variable(name='last_output', shape=[1, input_dim], initializer="random_normal", trainable = True)
        self.Wf = ErrorLinearLayer(input_dim * 2, input_dim)
        self.Wi = ErrorLinearLayer(input_dim * 2, input_dim)
        self.Wc = ErrorLinearLayer(input_dim * 2, input_dim)
        self.Wo = ErrorLinearLayer(input_dim * 2, input_dim)
        self.dims = input_dim

    def call(self, inputs):
        concat = self.kernel.random_error(tf.concat([self.h, inputs],1), ERROR_RATE, FLIP_START, FLIP_END)
        z = self.kernel.random_error(tf.keras.activations.tanh(self.Wc(concat)), ERROR_RATE, FLIP_START, FLIP_END)
        zf = self.kernel.random_error(tf.keras.activations.sigmoid(self.Wf(concat)), ERROR_RATE, FLIP_START, FLIP_END)
        zi = self.kernel.random_error(tf.keras.activations.sigmoid(self.Wi(concat)), ERROR_RATE, FLIP_START, FLIP_END)
        zo = self.kernel.random_error(tf.keras.activations.sigmoid(self.Wo(concat)), ERROR_RATE, FLIP_START, FLIP_END)
        memory = self.kernel.random_error(z * zi, ERROR_RATE, FLIP_START, FLIP_END)
        self.c = self.kernel.random_error(self.kernel.random_error(self.c * zf, ERROR_RATE, FLIP_START, FLIP_END) + memory, ERROR_RATE, FLIP_START, FLIP_END)
        self.h = self.kernel.random_error(zo * self.kernel.random_error(tf.keras.activations.tanh(self.c), ERROR_RATE, FLIP_START, FLIP_END), ERROR_RATE, FLIP_START, FLIP_END)
        return self.h

class ErrorLSTM(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim):
        super(ErrorLSTM, self).__init__()
        self.linear = ErrorLinearLayer(input_dim, hidden_dim)
        self.cell = ErrorLSTMCell(hidden_dim)

    def call(self, inputs):
        inputs = self.linear(inputs)
        for i in range(len(inputs)):
            for data in inputs[i]:
                output = self.cell([data])
            if i == 0:
                outputs = output
            else:
                outputs = tf.concat((outputs, output),0)
        return outputs

if __name__ == "__main__":
    model = LSTM(6,12)
    data = tf.constant([[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]],[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]]])
    print(data.shape)
    print(model(data))
    model = QuantLSTM(6,12)
    data = tf.constant([[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]],[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]]])
    print(data.shape)
    print(model(data))
    skrms = SKRM()
    model = SKRMLSTM(6,12,skrms)
    data = tf.constant([[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]],[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]]])
    print(data.shape)
    print(model(data))
    model = ErrorLSTM(6, 12)
    print(model(data))
