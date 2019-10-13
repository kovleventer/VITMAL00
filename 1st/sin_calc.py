from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
import keras.losses
from keras.optimizers import SGD, adam
import numpy as np
import random
import math
import matplotlib.pyplot as plt

X = np.arange(10000) / 1000 * math.pi
Y = np.sin(X)

model = Sequential()
model.add(Dense(100, input_shape=(1,), activation="relu"))
model.add(Dense(1))

model.compile(loss='mse', optimizer=SGD())

model.fit(X, Y, validation_split=.2, epochs=10)

X_test = np.arange(2000) / 1000 * math.pi + 1000
Y_test = np.sin(X_test)

#plt.plot(X_test, Y_test)
Y_res = model.predict(X)
plt.plot(X, Y_res)
plt.show()

score = model.evaluate(X_test, Y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

