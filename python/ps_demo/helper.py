import numpy as np
import kvstore_pb2

def np_dtype_to_dtype(np_dtype):
    if np_dtype == np.float16:
        return kvstore_pb2.Tensor.FP16
    elif np_dtype == np.float32:
        return kvstore_pb2.Tensor.FP32
    elif np_dtype == np.float64:
        return kvstore_pb2.Tensor.FP64
    elif np_dtype == np.int16:
        return kvstore_pb2.Tensor.INT16
    elif np_dtype == np.int32:
        return kvstore_pb2.Tensor.INT32
    elif np_dtype == np.int64:
        return kvstore_pb2.Tensor.INT64
    elif np_dtype == np.bool:
        return kvstore_pb2.Tensor.BOOL 
    else:
        raise ValueError("Not supported numpy dtype %s" % np_dtype)

def dtype_to_np_dtype(dtype):
    if dtype == kvstore_pb2.Tensor.FP16:
        return np.float16
    elif dtype == kvstore_pb2.Tensor.FP32:
        return np.float32
    elif dtype == kvstore_pb2.Tensor.FP64:
        return np.float64
    elif dtype == kvstore_pb2.Tensor.INT16:
        return np.int16
    elif dtype == kvstore_pb2.Tensor.INT32:
        return np.int32
    elif dtype == kvstore_pb2.Tensor.INT64:
        return np.int64
    elif dtype == kvstore_pb2.Tensor.BOOL:
        return np.bool
    else:
        raise ValueError("Not supported dtype %s" % dtype)

def size_of_dtype(dtype):
    if dtype == kvstore_pb2.Tensor.FP16:
        return 2
    elif dtype == kvstore_pb2.Tensor.FP32:
        return 4
    elif dtype == kvstore_pb2.Tensor.FP64:
        return 8
    elif dtype == kvstore_pb2.Tensor.INT16:
        return 2
    elif dtype == kvstore_pb2.Tensor.INT32:
        return 4
    elif dtype == kvstore_pb2.Tensor.INT64:
        return 8
    elif dtype == kvstore_pb2.Tensor.BOOL:
        return 1
    else:
        raise ValueError("Not supported dtype %s" % dtype)

def ndarray_to_tensor(name, arr):
    tensor = kvstore_pb2.Tensor()
    tensor.name = name
    tensor.dim.extend(arr.shape)
    tensor.content = arr.tobytes()
    tensor.data_type = np_dtype_to_dtype(arr.dtype)     
    return tensor

def tensor_to_ndarray(tensor):
    size = size_of_dtype(tensor.data_type)
    for d in tensor.dim:
        size *= d

    if size != len(tensor.content):
        raise ValueError(
            "Tensor  size mismatch, dim: %s, len(content): %d",
            tensor.dim,
            len(tensor.content),
        )

    arr = np.ndarray(
            shape=tensor.dim,
            dtype=dtype_to_np_dtype(tensor.data_type),
            buffer=tensor.content
        )
    return arr