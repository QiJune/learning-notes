import argparse
import time

from ps_mini.proto import core_pb2
from ps_mini.proto import core_pb2_grpc
import grpc
import numpy as np
from ps_mini.tensor import Tensor, deserialize_from_pb, serialize_to_pb


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
            stub = core_pb2_grpc.PServerStub(channel)
            self.pserver_stubs.append(stub)

    def pull_embedding_param(self, name, ids):
        tensor = Tensor(name, None, ids)
        pb = core_pb2.Tensor()
        serialize_to_pb(tensor, pb)
        res = self.pserver_stubs[0].pull_embedding_param(pb)
        res_tensor = Tensor()
        deserialize_from_pb(res, res_tensor)
        return res_tensor

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

    init_param = Tensor(name="tom",
                        value=np.ones(shape=(3, )),
                        initializer="uniform")
    worker.push_embedding_param(init_param)

    ids = [0, 1, 2, 3, 4, 5]
    res = worker.pull_embedding_param("tom", ids)
    print(res.value)
    print(res.indices)

    time.sleep(2)

    ids = [0, 3, 3]
    value = np.full((3, 3), 2.0)
    grad = Tensor("tom", value, ids)

    worker.push_embedding_grad(grad)

    time.sleep(2)

    ids = [0, 3, 7]
    res = worker.pull_embedding_param("tom", ids)
    print(res.value)
    print(res.indices)
