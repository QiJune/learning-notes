from embedding_table import EmbeddingTable
from google.protobuf import empty_pb2
from ps_mini.core_pb2 import EmbeddingResponse
from ps_mini.core_pb2 import Tensor as TensorPB
from ps_mini.core_pb2_grpc import KVStoreServicer
from ps_mini.tensor import Tensor, deserialize_from_pb, serialize_to_pb


class KVStore(KVStoreServicer):
    def __init__(self):
        self.param_db = {}
        self.embedding_param_db = {}

    def pull_param(self, request, _):
        tensor = self.param_db[request.name]
        pb = TensorPB()
        serialize_to_pb(tensor, pb)
        return pb

    def push_param(self, request, _):
        tensor = Tensor()
        deserialize_from_pb(request, tensor)
        self.param_db[request.name] = tensor
        return empty_pb2.Empty()

    def pull_embedding_param(self, request, _):
        response = EmbeddingResponse()
        if request.name not in self.embedding_param_db:
            embedding_param = EmbeddingTable(request.name, 0)
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
            embedding_param = EmbeddingTable(request.name, 0)
            self.embedding_param_db[request.name] = embedding_param
        embedding_param = self.embedding_param_db[tensor.name]
        embedding_param.set(tensor.indices, tensor.value)
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
