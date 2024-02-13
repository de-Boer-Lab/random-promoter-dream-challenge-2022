import numpy as np
import h5py
from scipy.stats import pearsonr

with h5py.File(('seq_onehot.h5'), 'r') as hf:
    X = hf['seq_onehot'][:]

with h5py.File(('expression.h5'), 'r') as hf:
    Y = hf['expression'][:]

from sklearn.model_selection import train_test_split

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.015)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Bidirectional, LSTM, \
    MultiHeadAttention

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
tpu_strategy = tf.distribute.TPUStrategy(tpu)

with tpu_strategy.scope():
    input_layer = tf.keras.layers.Input(shape=X_train.shape[1:])

    y = Conv1D(1000, 30, strides=1, activation="relu")(input_layer)
    y = MaxPooling1D(pool_size=3, strides=3)(y)
    y = Dropout(0.2)(y)
    y = Bidirectional(LSTM(320, return_sequences=True,kernel_initializer='he_normal'))(y)
    y = Dropout(0.2)(y)
    y = Flatten()(y)
    y = Dense(64, activation='relu')(y)
    y = Dense(64, activation='relu')(y)

    activation_layer = tf.keras.layers.Activation("relu")(y)

    output = tf.keras.layers.Dense(1)(activation_layer)

    # compile the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output, name="conv_model")
    model.compile(loss="mean_squared_error", optimizer="adam")

print(model.summary())

model.fit(X_train, y_train, batch_size=1024, epochs=10)

predictions = model.predict(X_test)
y_test_squeezed = np.squeeze(np.asarray(y_test))
predictions_squeezed = np.squeeze(np.asarray(predictions))
print(pearsonr(y_test_squeezed, predictions_squeezed))

model.save('model.h5')
