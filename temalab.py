# path
import os
from os.path import isdir, join
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers
from guppy import hpy
import random
import copy
import librosa
import time
import math

h = hpy()

train_audio_path = 'train/train/audio/'
target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

OUTPUTS = len(target_list) + 2

INPUT_SAMPLES = 8192
FFT_SHAPE = INPUT_SAMPLES // 2 + 1
DHT_SHAPE = INPUT_SAMPLES
NOISE_MULTIPLIER = 1 # How many noised copies should we make
UNKNOWN_COUNT = 2000 * (NOISE_MULTIPLIER + 1)
SILENCE_COUNT = 2000 * (NOISE_MULTIPLIER + 1)
NOISE_RATIO = .2

VALID_SPLIT = .2


def label_to_id(label):
    if label in target_list:
        return target_list.index(label)
    elif label == "unknown":
        return len(target_list)
    elif label == "silence":
        return len(target_list)+1

def shuffle(*arrays):
    state = np.random.get_state()
    for array in arrays:
        np.random.set_state(state)
        np.random.shuffle(array)

DHT_MTX = None
def dht(array):
    global DHT_MTX
    if DHT_MTX is None:
        N = len(array)
        ts = np.arange(N) / N
        fs = np.arange(N)
        args = np.outer(ts, fs)
        DHT_MTX = np.cos(2 * math.pi * args) + np.sin(2 * math.pi * args)
    return np.dot(DHT_MTX, array)

def generate_convolution(size):
    input = Input(shape=(size, 1))
    x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides='1')(input)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides='1')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides='1')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop_out_rate)(x)
    x = layers.Flatten()(x)
    return input, x


def load_data_from_wavs():

    background = [f for f in os.listdir(join(train_audio_path, '_background_noise_')) if f.endswith('.wav')]
    background_noise = []
    for wav in background:
        samples, sample_rate = librosa.load(join(join(train_audio_path, '_background_noise_'), wav), sr=INPUT_SAMPLES)
        background_noise.append(samples)

    dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
    dirs.sort()

    wavs = []
    labels = []
    unknown_wavs = []

    unknown_list = [d for d in dirs if d not in target_list and d != '_background_noise_']
    print('target_list : ', end='')
    print(target_list)
    print('unknowns_list : ', end='')
    print(unknown_list)
    print('silence : _background_noise_')

    i = 0
    for directory in dirs[1:]:
        start = time.time()
        waves = [f for f in os.listdir(join(train_audio_path, directory)) if f.endswith('.wav')]
        i = i + 1
        print(str(i) + ":" + str(directory) + " ", end="", flush=True)
        for wav in waves:
            samples, sample_rate = librosa.load(join(join(train_audio_path, directory), wav), sr=16000)
            samples = np.concatenate((np.zeros((INPUT_SAMPLES-8000)//2, dtype="float32"),
                                     samples[::2],
                                     np.zeros((INPUT_SAMPLES-8000)//2, dtype="float32")))
            # samples1 = librosa.resample(samples, sample_rate, 8000)
            # retek = samples2 - samples1
            if len(samples) != INPUT_SAMPLES:
                continue



            if directory in unknown_list:
                unknown_wavs.append(samples)
            else:
                wavs.append(samples)
                labels.append(directory)
        end = time.time()
        print(end - start, flush=True)

    cnt = len(wavs) * (NOISE_MULTIPLIER + 1) + UNKNOWN_COUNT + SILENCE_COUNT
    X = np.zeros((cnt, INPUT_SAMPLES), dtype="float32")
    X_fft = np.zeros((cnt, FFT_SHAPE), dtype="float32")
    X_dht = np.zeros((cnt, DHT_SHAPE), dtype="float32")
    Y = np.zeros((cnt, 1), dtype="int32")

    def get_one_noise():
        noise_num = random.randrange(0, len(background_noise))
        selected_noise = background_noise[noise_num]
        start_idx = random.randint(0, len(selected_noise) - 1 - INPUT_SAMPLES)
        return selected_noise[start_idx:(start_idx + INPUT_SAMPLES)]


    for i in range(len(wavs)):
        X[i] = wavs[i]
        Y[i] = label_to_id(labels[i])
    acc = len(wavs)

    for n in range(NOISE_MULTIPLIER):
        for i in range(len(wavs)):
            X[acc + n * len(wavs) + i] = get_one_noise() * NOISE_RATIO + wavs[i]
            Y[acc + n * len(wavs) + i] = label_to_id(labels[i])
    acc += len(wavs) * NOISE_MULTIPLIER

    for i in range(UNKNOWN_COUNT):
        X[acc + i] = random.choice(unknown_wavs)
        Y[acc + i] = label_to_id("unknown")
    acc += UNKNOWN_COUNT

    for i in range(SILENCE_COUNT):
        X[acc + i] = get_one_noise() * .5
        Y[acc + i] = label_to_id("silence")


    for i in range(cnt):
        X_fft[i] = np.abs(np.fft.rfft(X[i]))
        #X_dht[i] = dht(X[i])



    print(h.heap())
    return X, X_fft, X_dht, Y


xfile = "x.npy"
xfftfile = "xfft.npy"
xdhtfile = "xdht.npy"
yfile = "y.npy"
try:
    X = np.load(xfile)
    X_fft = np.load(xfftfile)
    X_dht = np.load(xdhtfile)
    Y = np.load(yfile)
    print("Loading arrays from file")
except:
    X, X_fft, X_dht, Y = load_data_from_wavs()
    np.save(xfile, X)
    np.save(yfile, Y)
    np.save(xdhtfile, X_dht)
    np.save(xfftfile, X_fft)


shuffle(X, X_fft, X_dht, Y)

# Parameters
lr = 0.001
generations = 20000
batch_size = 256
drop_out_rate = 0.5

# For Conv1D add Channel
X = X.reshape((-1, INPUT_SAMPLES, 1))
X_fft = X_fft.reshape((-1, FFT_SHAPE, 1))
X_dht = X_dht.reshape((-1, DHT_SHAPE, 1))


# Make Label data 'class num' -> 'One hot vector'
Y = keras.utils.to_categorical(Y, len(target_list)+2)


# Conv1D Model
input_x = Input(shape=(INPUT_SAMPLES, 1))
x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Flatten()(x)

input_xfft, x_fft = generate_convolution(DHT_SHAPE)

x = keras.layers.Concatenate()([x, x_fft])

x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
output_tensor = layers.Dense(OUTPUTS, activation='softmax')(x)

model = tf.keras.Model(inputs=[input_x, input_xfft
                               ], outputs=[output_tensor,])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=lr),
              metrics=['accuracy'])


model.summary()


history = model.fit([X, X_dht
                      ], Y,
                    validation_split=VALID_SPLIT,
                    batch_size=batch_size,
                    epochs=100,
                    verbose=1)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
