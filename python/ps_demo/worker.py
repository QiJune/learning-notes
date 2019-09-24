import grpc
import numpy as np

import kvstore_pb2
import kvstore_pb2_grpc

import time

from helper import np_dtype_to_dtype, dtype_to_np_dtype, size_of_dtype
from helper import ndarray_to_tensor, tensor_to_ndarray

# open a gRPC channel
channel = grpc.insecure_channel('localhost:50051')

# create a stub (client)
stub = kvstore_pb2_grpc.KVStoreStub(channel)

# create a valid request message
param = kvstore_pb2.Tensor()
param.name = "tom"
param.dim.extend((3,))
param.data_type = kvstore_pb2.Tensor.FP32
param = stub.pull_or_init_param(param)
param = tensor_to_ndarray(param)

arr = np.array([1, 2, 3], dtype="float32")
grad = ndarray_to_tensor("tom@grad", arr)
stub.push_grad(grad)

time.sleep(1)

new_param = kvstore_pb2.Tensor()
new_param.name = "tom"
res = tensor_to_ndarray(stub.pull_param(new_param))
print(res)
