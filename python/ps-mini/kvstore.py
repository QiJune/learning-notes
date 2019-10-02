import queue

import core_pb2
import core_pb2_grpc
import numpy as np
from google.protobuf import empty_pb2
from tensor import Tensor, deserialize_from_pb, serialize_to_pb


class EmbeddingParam(object):
    def __init__(self, name, version=None):
        self.name = name
        self.version = version
        self.db = {}

    def get(self, ids):
        values = []
        know_ids = []
        unknow_ids = []

        for id in ids:
            if id not in self.db:
                unknow_ids.append(id)
            else:
                value = self.db[id]
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
            self.db[id] = row


class KVStoreServicer(core_pb2_grpc.KVStoreServicer):
    def __init__(self):
        self.param_db = {}
        self.embedding_param_db = {}
        self.grads = queue.Queue()

    def pull_param(self, request, _):
        tensor = self.param_db[request.name]
        pb = tensor.write_to_pb()
        return pb

    def push_param(self, request, _):
        tensor = core_pb2.Tensor()
        deserialize_from_pb(request, tensor)
        self.param_db[request.name] = tensor
        return empty_pb2.Empty()

    def push_grad(self, request, _):
        tensor = core_pb2.Tensor()
        deserialize_from_pb(request, tensor)
        self.grads.put(tensor)
        return empty_pb2.Empty()

    def pull_embedding_param(self, request, _):
        response = core_pb2.EmbeddingResponse()
        if request.name not in self.embedding_param_db:
            embedding_param = EmbeddingParam(request.name, 0)
            self.embedding_param_db[request.name] = embedding_param
        embedding_param = self.embedding_param_db[request.name]
        ids = request.indices
        tensor, unknown_ids = embedding_param.get(ids)

        serialize_to_pb(tensor, response.value)
        response.unknown_indices.extend(unknown_ids)
        return response

    def push_embedding_param(self, request, _):
        tensor = Tensor()
        deserialize_from_pb(request, tensor)
        if request.name not in self.embedding_param_db:
            embedding_param = EmbeddingParam(request.name, 0)
            self.embedding_param_db[request.name] = embedding_param
        embedding_param = self.embedding_param_db[tensor.name]
        embedding_param.set(tensor.indices, tensor.value)
        return empty_pb2.Empty()

    def push_embedding_grad(self, request, _):
        tensor = Tensor()
        deserialize_from_pb(request, tensor)
        self.grads.put(tensor)
        return empty_pb2.Empty()

    def get_param(self, name):
        return self.param_db[name]

    def set_param(self, name, value):
        self.param_db[name] = value

    def get_embedding_param(self, name, ids):
        param, _ = self.embedding_param_db[name].get(ids)
        return param

    def set_embedding_param(self, name, ids, value):
        self.embedding_param_db[name].set(ids, value)
