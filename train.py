import tensorflow as tf 
import time

def Train(model, dataset, answer, num_epochs):
    print('Training...')
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2)  
    mse = tf.keras.losses.MeanSquaredError()
    metrics = tf.keras.metrics.Accuracy()
    startTime = time.time()
    total = num_epochs
    for x in range(num_epochs):
        with tf.GradientTape() as tape:
            y_pred = model(dataset)
            loss = mse(answer, y_pred)
            print(loss.numpy())
            metrics.update_state(answer, y_pred)       
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        print(f'epoch:{x} accuracy:{metrics.result().numpy()}')
        metrics.reset_states()
    print(f'cost time: {round(time.time() - startTime,3)} sec')