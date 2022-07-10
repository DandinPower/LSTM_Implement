import tensorflow as tf 
import numpy as np 
@ops.RegisterGradient("BitsQuant")
def _bits_quant_grad(op, grad):
    inputs = op.inputs[0]
    return [grad] 
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.w = self.add_variable(name='w',
            shape=[input_dim, output_dim], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b',
            shape=[output_dim], initializer=tf.zeros_initializer())

    def call(self, inputs):
        matmul = tf.matmul(inputs, self.w)
        bias = matmul + self.b
        return bias

class QuantLinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.kernel = tf.load_op_library('./bits_quant.so')
        self.w = self.add_variable(name='w',
            shape=[input_dim, output_dim], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b',
            shape=[output_dim], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = self.kernel.bits_quant(tf.matmul(inputs, self.w)) + self.kernel.bits_quant(self.b)
        return self.kernel.bits_quant(y_pred)

class LSTMCell(tf.keras.Model):
    def __init__(self, input_dim):
        super(LSTMCell, self).__init__()
        self.c = self.add_variable(name='memory_cell', shape=[1, input_dim], initializer=tf.zeros_initializer())
        self.h = self.add_variable(name='last_output', shape=[1, input_dim], initializer=tf.zeros_initializer())
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

class QuantLSTMCell(tf.keras.Model):
    def __init__(self, input_dim):
        super(QuantLSTMCell, self).__init__()
        self.kernel = tf.load_op_library('./bits_quant.so')
        self.c = self.add_variable(name='memory_cell', shape=[1, input_dim], initializer=tf.zeros_initializer())
        self.h = self.add_variable(name='last_output', shape=[1, input_dim], initializer=tf.zeros_initializer())
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

if __name__ == "__main__":
    model = LSTM(6,12)
    data = tf.constant([[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]],[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]]])
    print(data.shape)
    print(model(data))
    model = QuantLSTM(6,12)
    data = tf.constant([[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]],[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]]])
    print(data.shape)
    print(model(data))