import sys

from tensorflow import keras

import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(1)
tf.random.set_seed(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from keras import regularizers

if len(sys.argv) != 4:
    print("Insufficient arguments")
    sys.exit()

model_file_path = sys.argv[1]
train_epoch = sys.argv[2]
train_file_path = sys.argv[3]

#학습 데이터
df2 = pd.read_csv(train_file_path)
df2 = df2[['TIME', 'CH1']]
df2['TIME'] = pd.to_numeric(df2['TIME'], downcast='signed')
print(df2['TIME'].min(), df2['TIME'].max())

train = df2

print(train.shape)

TIME_STEPS=30

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])

    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(train[['CH1']], train['CH1'])

print(f'Training shape: {X_train.shape}')

model = Sequential()
model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=regularizers.l2(0.00), return_sequences=False))
model.add(Dropout(rate=0.3))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(rate=0.3))
model.add(TimeDistributed(Dense(X_train.shape[2])))
opt = keras.optimizers.Adam()
model.compile(optimizer=opt, loss='mae')
model.summary()

history = model.fit(X_train, y_train, batch_size=32, epochs=int(train_epoch), validation_split=0.2, shuffle=False)

model.save(model_file_path)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

