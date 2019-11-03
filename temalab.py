# path
import os
from os.path import isdir, join
from pathlib import Path

# Scientific Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.offline as py
import plotly.graph_objs as go

# Deep learning
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, layers
from tensorflow.keras import backend as K

import random
import copy
import librosa


train_audio_path = 'train/train/audio/'
target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

INPUT_SAMPLES = 8192
NOISE_MULTIPLIER = 1 # How many noised copies should we make
UNKNOWN_COUNT = 2000 * (NOISE_MULTIPLIER + 1)
SILENCE_COUNT = 2000 * (NOISE_MULTIPLIER + 1)

def load_waws():
    dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
    dirs.sort()
    print('Number of labels: ' + str(len(dirs[1:])))
    print(dirs)


    wavs = []
    labels = []
    unknown_wavs = []

    unknown_list = [d for d in dirs if d not in target_list and d != '_background_noise_']
    print('target_list : ', end='')
    print(target_list)
    print('unknowns_list : ', end='')
    print(unknown_list)
    print('silence : _background_noise_')

    import time

    i = 0
    for directory in dirs[1:]:
        start = time.time()
        waves = [f for f in os.listdir(join(train_audio_path, directory)) if f.endswith('.wav')]
        i = i + 1
        print(str(i) + ":" + str(directory) + " ", end="", flush=True)
        for wav in waves:
            samples, sample_rate = librosa.load(join(join(train_audio_path, directory), wav), sr=16000)
            samples = samples[::2]
            samples = np.concatenate((np.zeros((INPUT_SAMPLES-8000)//2),
                                     samples,
                                     np.zeros((INPUT_SAMPLES-8000)//2)))
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

    return wavs, labels, unknown_wavs

def load_background_wav():
    background = [f for f in os.listdir(join(train_audio_path, '_background_noise_')) if f.endswith('.wav')]
    background_noise = []
    for wav in background:
        samples, sample_rate = librosa.load(join(join(train_audio_path, '_background_noise_'), wav))
        print("asd ", sample_rate)
        samples = librosa.resample(samples, sample_rate, INPUT_SAMPLES)
        background_noise.append(samples)
    return background_noise


wavs, labels, unknown_wavs = load_waws()
background_noise = load_background_wav()

wavs = np.array(wavs)
labels = np.array(labels)


# Random pick start point
def get_one_noise(noise_num=0):
    selected_noise = background_noise[noise_num]
    start_idx = random.randint(0, len(selected_noise) - 1 - INPUT_SAMPLES)
    return selected_noise[start_idx:(start_idx + INPUT_SAMPLES)]


max_ratio = 0.1
noised_wavs = []
NOISE_MULTIPLIER = 1
for i in range(NOISE_MULTIPLIER):
    new_wav = []
    noise = get_one_noise(i)
    for i, s in enumerate(wavs):
        s = s + (max_ratio * noise)
        noised_wavs.append(s)




labels_orig = copy.deepcopy(labels)
for _ in range(NOISE_MULTIPLIER):
    labels = np.concatenate((labels, labels_orig), axis=0)
labels = labels.reshape(-1, 1)



np.random.shuffle(unknown_wavs)
unknown_wavs = np.array(unknown_wavs)
unknown_wavs = unknown_wavs[:UNKNOWN_COUNT]
unknown_labels = np.array(['unknown' for _ in range(UNKNOWN_COUNT)])
unknown_labels = unknown_labels.reshape(-1, 1)



# silence audio
silence_wavs = []
num_wav = SILENCE_COUNT // len(background_noise)
for i, _ in enumerate(background_noise):
    for _ in range(SILENCE_COUNT // len(background_noise)):
        silence_wavs.append(get_one_noise(i))
silence_wavs = np.array(silence_wavs)
silence_labels = np.array(['silence' for _ in range(num_wav * len(background_noise))])
silence_labels = silence_labels.reshape(-1, 1)


noised_wavs = np.reshape(noised_wavs, (-1, INPUT_SAMPLES))



print(wavs.shape)
print(noised_wavs.shape)
print(unknown_wavs.shape)
print(silence_wavs.shape)

# %%

print(labels.shape)
print(unknown_labels.shape)
print(silence_labels.shape)



wavs = np.concatenate((wavs, noised_wavs, unknown_wavs, silence_wavs), axis=0)
labels = np.concatenate((labels, unknown_labels, silence_labels), axis=0)

print(wavs.shape)
print(labels.shape)


train_wav, test_wav, train_label, test_label = train_test_split(wavs, labels,
                                                                test_size=0.2,
                                                                random_state=1993,
                                                                shuffle=True)
wavs = None
silence_labels = None
labels = None
unknown_labels = None
unknown_wavs = None
silence_wavs = None
noised_wavs = None
import gc
gc.collect()

# Parameters
lr = 0.001
generations = 20000
num_gens_to_wait = 250
batch_size = 512
drop_out_rate = 0.5
input_shape = (INPUT_SAMPLES, 1)

# %%

# For Conv1D add Channel
train_wav = train_wav.reshape(-1, INPUT_SAMPLES, 1)
test_wav = test_wav.reshape(-1, INPUT_SAMPLES, 1)

# %%

label_value = target_list
label_value.append('unknown')
label_value.append('silence')

# %%

new_label_value = dict()
for i, l in enumerate(label_value):
    new_label_value[l] = i
label_value = new_label_value

# %%

# Make Label data 'string' -> 'class num'
temp = []
for label in train_label:
    temp.append(label_value[label[0]])
train_label = np.array(temp)

temp = []
for label in test_label:
    temp.append(label_value[label[0]])
test_label = np.array(temp)

# Make Label data 'class num' -> 'One hot vector'
train_label = keras.utils.to_categorical(train_label, len(label_value))
test_label = keras.utils.to_categorical(test_label, len(label_value))


# Conv1D Model
input_tensor = Input(shape=input_shape)

x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_tensor)
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
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(drop_out_rate)(x)
output_tensor = layers.Dense(len(label_value), activation='softmax')(x)

model = tf.keras.Model(input_tensor, output_tensor)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=lr),
              metrics=['accuracy'])

# %%

model.summary()


history = model.fit(train_wav, train_label, validation_data=[test_wav, test_label],
                    batch_size=batch_size,
                    epochs=100,
                    verbose=1)

# %%

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
