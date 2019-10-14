from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
import keras.losses
from keras.optimizers import SGD, adam
import numpy as np
import random
import math
import matplotlib.pyplot as plt


INPUT = 40
CNT = 1000

X_pre = np.arange(INPUT + CNT+1) / CNT * 2 * math.pi
Y_pre = np.sin(X_pre)


X = np.zeros((CNT, INPUT))
Y = np.zeros((CNT,))
for i in range(CNT):
    X[i] = Y_pre[i:i+INPUT]
    Y[i] = Y_pre[i+INPUT+1]

model = Sequential()
model.add(Dense(100, input_shape=(INPUT,)))
model.add(LeakyReLU())
model.add(Dense(100, activation="relu"))

model.add(Dense(1))

model.compile(loss='mse', optimizer=SGD())

model.fit(X, Y, validation_split=.2, epochs=60)

X_pred = np.arange(INPUT) / CNT * 2 * math.pi
Y_pred = np.sin(X_pred)

X_l = list(X_pred)
Y_l = list(Y_pred)
for i in range(10000):
    Y_new = model.predict(np.array(Y_l[-INPUT:]).reshape(1, -1))
    Y_l.append(Y_new)
    X_l.append(X_l[-1] + 1/CNT*2*math.pi)


plt.plot(X_l, Y_l)
plt.show()
