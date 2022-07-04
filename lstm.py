import tensorflow as tf 
import numpy as np 

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

if __name__ == "__main__":
    model = LSTM(6,12)
    data = tf.constant([[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]],[[0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6], [0.1,0.2,0.3,0.4,0.5,0.6]]])
    print(data.shape)
    print(model(data))