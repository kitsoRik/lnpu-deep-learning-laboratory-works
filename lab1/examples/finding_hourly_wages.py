import pandas as pd
import tensorflow as tf

# load data
train_X = pd.read_csv('../datasets/hourly_wages_data.csv')
train_Y = train_X.pop('manufacturing')

# setup model
model = tf.keras.Sequential()

n_cols = train_X.shape[1]

# add layers
model.add(tf.keras.layers.Dense(200, activation='relu', input_shape=(n_cols,)))
model.add(tf.keras.layers.Dense(200, activation='relu'))
model.add(tf.keras.layers.Dense(200, activation='relu'))
model.add(tf.keras.layers.Dense(1))

# compile
model.compile(optimizer='adam', loss='mean_squared_error')

# callback
early_stop_monitoring = tf.keras.callbacks.EarlyStopping(patience=3)

# training
model.fit(train_X, train_Y, validation_split=0.2, epochs=30, callbacks=[early_stop_monitoring])

# prediction
test_y_prediction = model.predict([[1, 1, 1, 1, 1, 1, 1, 1, 1]])

print(test_y_prediction)
