from google.protobuf import empty_pb2
import numpy as np

import core_pb2
import core_pb2_grpc

from helper import np_dtype_to_dtype, dtype_to_np_dtype, size_of_dtype
from helper import ndarray_to_tensor, tensor_to_ndarray

import queue
from helper import dtype_to_np_dtype
    
class KVStoreServicer(core_pb2_grpc.KVStoreServicer):
    def __init__(self):
        self.db = {}
        self.grads = queue.Queue()

    def update_db(self, kv_pair):
        self.db[kv_pair[0]] = kv_pair[1]

    def query_db(self, key):
        if key in self.db:
            return self.db[key]
        return None
    
    def pull_param(self, request, _):
        np_data = self.query_db(request.name)
        tensor = ndarray_to_tensor(request.name, np_data)
        return tensor

    def pull_or_init_param(self, request, _):
        np_data = self.query_db(request.name)
        if np_data is None:
            dim = request.dim
            dtype = dtype_to_np_dtype(request.data_type)
            # use np.ones temporily, will changed to customized initializer
            arr = np.ones(shape=dim, dtype=dtype)
            self.update_db((request.name, arr))
            tensor = ndarray_to_tensor(request.name, arr)      
        else:
            tensor = ndarray_to_tensor(request.name, np_data)
        return tensor

    def push_grad(self, request, _):
        cur_gradient = (request.name, tensor_to_ndarray(request))
        self.grads.put(cur_gradient)
        return empty_pb2.Empty()

    