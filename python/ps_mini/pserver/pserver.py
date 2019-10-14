import argparse
from concurrent import futures

import grpc
from gradient_queue import GradientQueue
from kvstore import KVStore
from optimizer import Optimizer
from ps_mini.core_pb2_grpc import (
    add_GradientQueueServicer_to_server,
    add_KVStoreServicer_to_server,
)
from tensorflow.keras.optimizers import SGD


class PServer(object):
    def __init__(self, opt):
        self.kvstore = KVStore()
        self.grad_queue = GradientQueue()
        self.opt = Optimizer(opt, self.grad_queue, self.kvstore)

    def start(self, endpoint):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_KVStoreServicer_to_server(self.kvstore, server)
        add_GradientQueueServicer_to_server(self.grad_queue, server)
        print("Starting server. Listening on endpoint %s." % endpoint)
        server.add_insecure_port(endpoint)
        server.start()
        try:
            while True:
                self.opt.apply_gradients()
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint", type=str)
    args = parser.parse_args()

    opt = SGD(lr=0.01)
    ps = PServer(opt)
    ps.start(args.endpoint)
