from wer import wer
from data_generator import AudioGenerator
from keras import backend as K
from utils import int_sequence_to_text
import numpy as np

from keras.callbacks import ModelCheckpoint, Callback

valid_cache = []
data_gen = AudioGenerator(spectrogram=True)
data_gen.load_train_data()
data_gen.load_validation_data()
for index in range(len(data_gen.valid_texts)):
    transcr = data_gen.valid_texts[index]
    audio_path = data_gen.valid_audio_paths[index]
    data_point = data_gen.normalize(data_gen.featurize(audio_path))
    valid_cache.append(data_point)


def calculate_wer2(input_to_softmax, model_path, words=False):
    # data_gen = AudioGenerator()
    # data_gen.load_train_data()
    # data_gen.load_validation_data()
    wers = []
    input_to_softmax.load_weights(model_path)

    l = len(data_gen.valid_texts)
    l = 100
    for index in range(l):
        transcr = data_gen.valid_texts[index]
        # audio_path = data_gen.valid_audio_paths[index]
        # data_point = data_gen.normalize(data_gen.featurize(audio_path))

        data_point = valid_cache[index]

        prediction = input_to_softmax.predict(np.expand_dims(data_point, axis=0))
        output_length = [input_to_softmax.output_length(data_point.shape[0])]
        pred_ints = (K.eval(K.ctc_decode(
            prediction, output_length)[0][0]) + 1).flatten().tolist()

        pred = ''.join(int_sequence_to_text(pred_ints))

        if words:
            w = wer(transcr.split(), pred.split())
        else:
            w = wer(list(transcr), list(pred))
        wers.append(w)
        if index % 100 == 0:
            print(index, len(data_gen.valid_texts), wers[-1])

    print("FINAL WER:", sum(wers) / len(wers), "words:", words)
    return sum(wers) / len(wers)


class WERCallback(Callback):
    def __init__(self, input_to_softmax, save_model_path):
        super().__init__()
        self.wercb = calculate_wer2
        self.its = input_to_softmax
        self.smp = save_model_path

    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 10 == 0:
        if True:
            print("EPOCH,", epoch)
            w = self.wercb(self.its, self.smp, True)
            c = self.wercb(self.its, self.smp, False)
            with open("basicrun.txt", "at") as f:
                f.write("" + str(w) + " " + str(c) + "\n")