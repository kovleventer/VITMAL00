from keras.engine import Layer
from keras import backend as K

class Swish(Layer):
    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return (inputs * K.sigmoid(self.beta * inputs))

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
