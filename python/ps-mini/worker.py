from google.protobuf import empty_pb2
import grpc
import numpy as np

import core_pb2
import core_pb2_grpc

import time

from concurrent import futures

from helper import np_dtype_to_dtype, dtype_to_np_dtype, size_of_dtype
from helper import ndarray_to_tensor, tensor_to_ndarray

import argparse

import tensorflow as tf

from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


class Worker(object):
    def __init__(self, master_endpoint, worker_id):
        self.master_endpoint = master_endpoint
        self.worker_id = worker_id
        self.master_channel = grpc.insecure_channel(self.master_endpoint)
        self.master_stub = core_pb2_grpc.MasterStub(self.master_channel)
        self.setup_pserver_stub()
        self.executor = futures.ThreadPoolExecutor(max_workers=10)
        
    def setup_pserver_stub(self):
        request = core_pb2.Worker()
        request.id = self.worker_id
        pserver_endpoints = self.master_stub.get_pserver(request)
        self.pserver_stubs = []
        for p in pserver_endpoints.endpoint:
            channel = grpc.insecure_channel(p)
            stub = core_pb2_grpc.KVStoreStub(channel)
            self.pserver_stubs.append(stub)

    def _get_stub(self, id):
        stub_num = len(self.pserver_stubs)
        pserver_id = id % stub_num
        return self.pserver_stubs[pserver_id]

    def pull_or_init_param(self, name, ids, dim):
        def impl(id):
            stub = self._get_stub(id)
            key = name + "_" + str(id)
            param = core_pb2.Tensor()
            param.name = key
            param.dim.extend(dim)
            param.data_type = core_pb2.Tensor.FP32
            param = stub.pull_or_init_param(param)
            param = tensor_to_ndarray(param)
            return param
        fs = [self.executor.submit(impl, id) for id in ids]
        res = [f.result() for f in fs]
        res = np.stack(res)
        return res

    def param_sharding(self):
        self.setup_network()
        for var in self.network.trainable_variables:
            print(var.name)


    def push_grad(self, grads):
        pass

    def pull_param(self):
        pass

    def push_param(self):
        pass
    
    def setup_network(self):
        self.network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(10)])
        self.network.build(input_shape=(None, 28*28))

    def forward_backward(self, x, y):
        self.pull_param()
        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28*28))
            # [b, 784] => [b, 10]
            out = self.network(x)
            # [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)
            # [b, 10]
            loss = tf.square(out - y_onehot)
            # [b]
            self.loss = tf.reduce_sum(loss) / 32
        grads = tape.gradient(self.loss, self.network.trainable_variables)
        self.push_grad(grads)

    def run(self):
        (xs, ys), _ = datasets.mnist.load_data()
        xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
        db = tf.data.Dataset.from_tensor_slices((xs, ys))
        db = db.batch(32).repeat(10)
        for step, (x,y) in enumerate(db):
            self.forward_backward(x, y)
            if step % 200 == 0:
                print(step, 'loss:', float(self.loss))

        
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint", type=str)
    parser.add_argument("-i", "--worker_id", type=int)
    args=parser.parse_args()

    worker = Worker(args.endpoint, args.worker_id)

    ids = [0, 1, 2, 3, 4, 5]
    res = worker.pull_or_init_param("tom", ids, (3,))
    for r in res:
        print(r)