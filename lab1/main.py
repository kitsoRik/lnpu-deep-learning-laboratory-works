import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# train_test_split - function that splits data
from sklearn.model_selection import train_test_split

# load data
raw_X = pd.read_csv('datasets/auto-mpg.csv')
raw_Y = raw_X.pop('mpg')

x_scaler = StandardScaler()
x_scaler.fit(raw_X)

raw_X = x_scaler.transform(raw_X)

# split data for train and test+val (0.1 + 0.2)
train_X, test_and_val_X, train_Y, test_and_val_Y = train_test_split(raw_X, raw_Y, test_size=0.3, shuffle=True)

# split data for test and val (0.2/0.3)
test_X, val_X, test_Y, val_Y = train_test_split(test_and_val_X, test_and_val_Y, test_size=0.66, shuffle=True)

# setup model
model = tf.keras.Sequential()

n_cols = train_X.shape[1]

# add layers
model.add(tf.keras.layers.Dense(72, activation='relu', input_shape=(n_cols,)))
model.add(tf.keras.layers.Dense(144, activation='relu', input_shape=(n_cols,)))
model.add(tf.keras.layers.Dense(72, activation='relu', input_shape=(n_cols,)))
model.add(tf.keras.layers.Dense(1))

# compile
model.compile(optimizer=tf.optimizers.Adam(
    learning_rate=0.001
), loss='mean_squared_error')

# training
history = model.fit(train_X, train_Y, validation_data=(val_X, val_Y.to_numpy()), epochs=1000, callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)
])

plt.title("Model loss (mse)")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Loss', 'Val loss'])
plt.show()

print("Evaluate loss:", model.evaluate(test_X, test_Y.to_numpy(), verbose=0))
