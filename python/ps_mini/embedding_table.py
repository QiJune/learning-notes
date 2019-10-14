import numpy as np
from tensor import Tensor


class EmbeddingTable(object):
    def __init__(self, name, version=None):
        self.name = name
        self.version = version
        self.vectors = {}

    def get(self, ids):
        values = []
        know_ids = []
        unknow_ids = []

        for id in ids:
            if id not in self.vectors:
                unknow_ids.append(id)
            else:
                value = self.vectors[id]
                values.append(value)
                know_ids.append(id)

        if len(values) == 0:
            tensor = Tensor(self.name, None, None, self.version)
        else:
            values = np.stack(values)
            tensor = Tensor(self.name, values, know_ids, self.version)

        return tensor, unknow_ids

    def set(self, ids, value):
        for index, id in enumerate(ids):
            row = value[index, :]
            self.vectors[id] = row
