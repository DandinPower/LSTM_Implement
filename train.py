import tensorflow as tf 
import time
import math

def Accuracy(y, y_pred):
    y_total = 0
    y_pred_total = 0
    for i in range(len(y)):
        y_total += abs(y[i][0])
        y_pred_total += abs(y_pred[i][0])
    return y_pred_total / y_total


def Train(model, dataset, answer, num_epochs):
    print('Training...')
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2)  
    mse = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanSquaredError()
    startTime = time.time()
    total = num_epochs
    for x in range(num_epochs):
        with tf.GradientTape() as tape:
            y_pred = model(dataset)
            loss = mse(answer, y_pred)
            print(loss)
            metrics.update_state(answer, y_pred)       
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        print(f'epoch:{x} loss:{metrics.result().numpy()} accuracy: {Accuracy(answer, y_pred)}')
        metrics.reset_states()
    print(f'cost time: {round(time.time() - startTime,3)} sec')