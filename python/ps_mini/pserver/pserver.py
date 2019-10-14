import argparse
from concurrent import futures

import grpc
from ps_mini.pserver.gradient_queue import GradientQueue
from ps_mini.pserver.kvstore import KVStore
from ps_mini.pserver.optimizer import Optimizer
from tensorflow.keras.optimizers import SGD
from google.protobuf import empty_pb2
from ps_mini.embedding_table import EmbeddingTable

from ps_mini.proto.core_pb2 import Tensor as TensorPB
from ps_mini.tensor import Tensor, serialize_to_pb, deserialize_from_pb
from ps_mini.proto.core_pb2_grpc import add_PServerServicer_to_server


class PServer(object):
    def __init__(self, opt):
        self.kvstore = KVStore()
        self.grad_queue = GradientQueue()
        self.opt = Optimizer(opt, self.grad_queue, self.kvstore)

    def pull_param(self, request, _):
        tensor = self.kvstore.get_param(request.name)
        pb = TensorPB()
        serialize_to_pb(tensor, pb)
        return pb

    def push_param(self, request, _):
        tensor = Tensor()
        deserialize_from_pb(request, tensor)
        self.kvstore.set_param(request.name, tensor)
        return empty_pb2.Empty()

    def pull_embedding_param(self, request, _):
        pb = TensorPB()
        if request.name not in self.kvstore.embedding_param_db:
            embedding_param = EmbeddingTable(request.name, 0, request.dim,
                                             request.initializer)
            self.kvstore.set_embedding_table(request.name, embedding_param)
        ids = request.indices
        tensor = self.kvstore.get_embedding_param(request.name, ids)
        serialize_to_pb(tensor, pb)
        return pb

    def push_embedding_param(self, request, _):
        tensor = Tensor()
        deserialize_from_pb(request, tensor)
        if request.name not in self.kvstore.embedding_param_db:
            embedding_param = EmbeddingTable(request.name, 0, request.dim,
                                             request.initializer)
            self.kvstore.set_embedding_table(request.name, embedding_param)
        return empty_pb2.Empty()

    def push_grad(self, request, _):
        tensor = Tensor()
        deserialize_from_pb(request, tensor)
        self.grad_queue.put_grad(tensor)
        return empty_pb2.Empty()

    def push_embedding_grad(self, request, _):
        tensor = Tensor()
        deserialize_from_pb(request, tensor)
        self.grad_queue.put_grad(tensor)
        return empty_pb2.Empty()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint", type=str)
    args = parser.parse_args()

    opt = SGD(lr=0.01)
    ps = PServer(opt)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_PServerServicer_to_server(ps, server)
    print("Starting pserver. Listening on endpoint %s." % args.endpoint)
    server.add_insecure_port(args.endpoint)
    server.start()
    try:
        while True:
            ps.opt.apply_gradients()
    except KeyboardInterrupt:
        server.stop(0)
