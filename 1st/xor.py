from keras.models import Sequential
from keras.layers import Dense
import keras.losses
from keras.optimizers import SGD
import numpy as np
import random

random.seed(123)

test_split = .1
valid_split = .2

N = 10000
X = np.zeros((N, 2))
Y = np.zeros((N, 1))

for n in range(N):
    r = random.randint(0, 3)

    xs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ys = [0, 1, 1, 0]
    X[n] = (xs[r])
    Y[n] = (ys[r])

X_test = X[:int(N * test_split)]
Y_test = Y[:int(N * test_split)]
X_valid = X[int(N * test_split):int(N * (test_split + valid_split))]
Y_valid = Y[int(N * test_split):int(N * (test_split + valid_split))]
X_train = X[int(N * test_split + valid_split):]
Y_train = Y[int(N * test_split + valid_split):]

model = Sequential()
model.add(Dense(10, input_shape=(2,), activation="relu"))
model.add(Dense(1, activation="relu"))
model.compile(loss=keras.losses.mean_squared_error, optimizer=SGD(), metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), nb_epoch=10)

score = model.evaluate(X_test, Y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
