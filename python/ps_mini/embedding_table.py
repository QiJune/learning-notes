from ps_mini.tensor import Tensor

import tensorflow as tf
import numpy as np


class EmbeddingTable(object):
    def __init__(self, name, version=None, dim=None, initializer=None):
        self.name = name
        self.version = version
        self.vectors = {}
        self.dim = dim
        self.initializer = initializer

    def get(self, ids):
        values = []
        for i in ids:
            if i not in self.vectors:
                initializer = tf.keras.initializers.get(self.initializer)
                value = initializer(shape=self.dim).numpy()
                self.vectors[i] = value
            else:
                value = self.vectors[i]
            values.append(value)

        if len(values) == 0:
            tensor = Tensor(self.name, None, None, self.version)
        else:
            values = np.stack(values)
            tensor = Tensor(self.name, values, ids, self.version)
        return tensor

    def set(self, ids, value):
        for index, id in enumerate(ids):
            row = value[index, :]
            self.vectors[id] = row
