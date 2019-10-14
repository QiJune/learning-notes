from ps_mini.proto import core_pb2
import numpy as np


def np_dtype_to_dtype(np_dtype):
    if np_dtype == np.float16:
        return core_pb2.Tensor.FP16
    elif np_dtype == np.float32:
        return core_pb2.Tensor.FP32
    elif np_dtype == np.float64:
        return core_pb2.Tensor.FP64
    elif np_dtype == np.int16:
        return core_pb2.Tensor.INT16
    elif np_dtype == np.int32:
        return core_pb2.Tensor.INT32
    elif np_dtype == np.int64:
        return core_pb2.Tensor.INT64
    elif np_dtype == np.bool:
        return core_pb2.Tensor.BOOL
    else:
        raise ValueError("Not supported numpy dtype %s" % np_dtype)


def dtype_to_np_dtype(dtype):
    if dtype == core_pb2.Tensor.FP16:
        return np.float16
    elif dtype == core_pb2.Tensor.FP32:
        return np.float32
    elif dtype == core_pb2.Tensor.FP64:
        return np.float64
    elif dtype == core_pb2.Tensor.INT16:
        return np.int16
    elif dtype == core_pb2.Tensor.INT32:
        return np.int32
    elif dtype == core_pb2.Tensor.INT64:
        return np.int64
    elif dtype == core_pb2.Tensor.BOOL:
        return np.bool
    else:
        raise ValueError("Not supported dtype %s" % dtype)


def size_of_dtype(dtype):
    if dtype == core_pb2.Tensor.FP16:
        return 2
    elif dtype == core_pb2.Tensor.FP32:
        return 4
    elif dtype == core_pb2.Tensor.FP64:
        return 8
    elif dtype == core_pb2.Tensor.INT16:
        return 2
    elif dtype == core_pb2.Tensor.INT32:
        return 4
    elif dtype == core_pb2.Tensor.INT64:
        return 8
    elif dtype == core_pb2.Tensor.BOOL:
        return 1
    else:
        raise ValueError("Not supported dtype %s" % dtype)


class Tensor(object):
    def __init__(self,
                 name=None,
                 value=None,
                 indices=None,
                 version=None,
                 initializer=None):
        self.name = name
        self.value = value
        self.indices = indices
        self.version = version
        self.initializer = initializer

    def debug_string(self):
        res = ""
        if self.name:
            res += "name: " + self.name + "\n"
        if self.value is not None:
            res += "value: " + str(self.value) + "\n"
        if self.indices is not None:
            res += "indices: " + str(self.indices)
        if self.initializer:
            res += "initializer: " + str(self.initializer) + "\n"
        if self.version:
            res += "version: " + str(self.version) + "\n"
        return res


def serialize_to_pb(tensor, pb):
    pb.name = tensor.name
    if tensor.value is not None:
        pb.dim.extend(tensor.value.shape)
        pb.data_type = np_dtype_to_dtype(tensor.value.dtype)
        pb.content = tensor.value.tobytes()
    if tensor.indices is not None:
        pb.indices.extend(tensor.indices)
    if tensor.version is not None:
        pb.version = tensor.version
    if tensor.initializer is not None:
        pb.initializer = tensor.initializer


def deserialize_from_pb(pb, tensor):
    if len(pb.content) > 0:
        size = size_of_dtype(pb.data_type)
        for d in pb.dim:
            size *= d

        if size != len(pb.content):
            raise ValueError(
                "Tensor  size mismatch, dim: %s, len(content): %d",
                pb.dim,
                len(pb.content),
            )

        arr = np.ndarray(
            shape=pb.dim,
            dtype=dtype_to_np_dtype(pb.data_type),
            buffer=pb.content,
        )
        tensor.value = arr
    tensor.name = pb.name
    tensor.indices = pb.indices
    tensor.version = pb.version
    tensor.initializer = pb.initializer
