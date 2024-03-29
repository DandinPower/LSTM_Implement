import tensorflow as tf 
import time
import math

def Accuracy(y, y_pred):
    total_accuracy = 0
    for i in range(len(y)):
        different = abs(y[i][0] - y_pred[i][0])
        accuracy = 1 - (different / y[i][0])
        total_accuracy += accuracy
    return total_accuracy / len(y)


def Train(model, dataset, answer, num_epochs):
    print('Training...')
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2)  
    mse = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanSquaredError()
    startTime = time.time()
    total = num_epochs
    history = []
    for x in range(num_epochs):
        with tf.GradientTape() as tape:
            y_pred = model(dataset)
            loss = mse(answer, y_pred)
            metrics.update_state(answer, y_pred)       
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        print(f'epoch:{x} loss:{metrics.result().numpy()} accuracy: {Accuracy(answer, y_pred)}')
        history.append(Accuracy(answer, y_pred).numpy())
        metrics.reset_states()
    print(f'cost time: {round(time.time() - startTime,3)} sec')
    return history

def Train_V2(logger, model, dataset, answer, num_epochs):
    print('Training...')
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2)  
    mse = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.MeanSquaredError()
    startTime = time.time()
    total = num_epochs
    history = []
    for x in range(num_epochs):
        logger.SetNewEpochs()
        with tf.GradientTape() as tape:
            y_pred = model(dataset)
            loss = mse(answer, y_pred)
            metrics.update_state(answer, y_pred)       
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        print(f'epoch:{x} loss:{metrics.result().numpy()} accuracy: {Accuracy(answer, y_pred)}')
        history.append(Accuracy(answer, y_pred).numpy())
        metrics.reset_states()
    print(f'cost time: {round(time.time() - startTime,3)} sec')
    return history

def WriteHistory(_history, _path):
    with open(_path, 'w') as f:
        for item in _history:
            f.write(f'{item}\n')