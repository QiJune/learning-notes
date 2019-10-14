import argparse
import time

import core_pb2
import core_pb2_grpc
import grpc
import numpy as np
import tensorflow as tf
from tensor import Tensor, deserialize_from_pb, serialize_to_pb


class KVStoreClient(object):
    def __init__(self, master_endpoint, worker_id):
        self.master_endpoint = master_endpoint
        self.worker_id = worker_id
        self.master_channel = grpc.insecure_channel(self.master_endpoint)
        self.master_stub = core_pb2_grpc.MasterStub(self.master_channel)
        self.setup_pserver_stub()

    def setup_pserver_stub(self):
        request = core_pb2.Worker()
        request.id = self.worker_id
        pserver_endpoints = self.master_stub.get_pserver(request)
        self.pserver_stubs = []
        for p in pserver_endpoints.endpoint:
            channel = grpc.insecure_channel(p)
            stub = core_pb2_grpc.KVStoreStub(channel)
            self.pserver_stubs.append(stub)

    def pull_embedding_param(self, name, ids, dim, initializer):
        tensor = Tensor(name, None, ids)
        pb = core_pb2.Tensor()
        serialize_to_pb(tensor, pb)
        res = self.pserver_stubs[0].pull_embedding_param(pb)
        know_ids = res.value.indices
        unknown_ids = res.unknown_indices

        if len(unknown_ids) == 0:
            tensor = Tensor()
            deserialize_from_pb(res.value, tensor)
            return tensor

        values = []
        for id in unknown_ids:
            initializer = tf.keras.initializers.get(initializer)
            tmp = initializer(shape=dim).numpy()
            values.append(tmp)
        value = np.stack(values)
        tensor = Tensor(name, value, unknown_ids)
        pb = core_pb2.Tensor()
        serialize_to_pb(tensor, pb)
        self.pserver_stubs[0].push_embedding_param(pb)
        res_new = self.pserver_stubs[0].pull_embedding_param(pb)
        unknown_ids_new = res_new.unknown_indices
        if len(unknown_ids_new) > 0:
            raise Exception("Update embedding vector failed")

        if len(know_ids) == 0:
            return tensor
        else:
            know_tensor = Tensor()
            deserialize_from_pb(res.value, know_tensor)
            know_tensor.indices.extend(tensor.indices)
            know_tensor.value = np.append(know_tensor.value, tensor.value)
            new_dim = (-1,) + dim
            know_tensor.value = np.reshape(know_tensor.value, new_dim)
            return know_tensor

    def push_embedding_param(self, param):
        pb = core_pb2.Tensor()
        serialize_to_pb(param, pb)
        self.pserver_stubs[0].push_embedding_param(pb)

    def push_embedding_grad(self, grad):
        pb = core_pb2.Tensor()
        serialize_to_pb(grad, pb)
        self.pserver_stubs[0].push_embedding_grad(pb)

    def pull_param(self, name):
        pb = core_pb2.Tensor()
        pb.name = name
        res = self.pserver_stubs[0].pull_param(pb)
        tensor = Tensor()
        deserialize_from_pb(res, tensor)
        return tensor

    def push_param(self, param):
        pb = core_pb2.Tensor()
        serialize_to_pb(param, pb)
        self.pserver_stubs[0].push_param(pb)

    def push_grad(self, grad):
        pb = core_pb2.Tensor()
        serialize_to_pb(grad, pb)
        self.pserver_stubs[0].push_grad(pb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint", type=str)
    parser.add_argument("-i", "--worker_id", type=int)
    args = parser.parse_args()

    worker = KVStoreClient(args.endpoint, args.worker_id)

    ids = [0, 1, 2, 3, 4, 5]
    res = worker.pull_embedding_param("tom", ids, (3,), "uniform")
    print(res.value)
    print(res.indices)

    ids = [0, 3, 3]
    value = np.full((3, 3), 2.0)
    grad = Tensor("tom", value, ids)

    worker.push_embedding_grad(grad)

    time.sleep(2)

    ids = [0, 3, 7]
    res = worker.pull_embedding_param("tom", ids, (3,), "uniform")
    print(res.value)
    print(res.indices)
