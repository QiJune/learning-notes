import argparse
from concurrent import futures

import core_pb2_grpc
import grpc
import tensorflow as tf
from kvstore_service import KVStoreServicer
from tensorflow.keras.optimizers import SGD


def convert_to_var(param):
    if param.indices:
        new_dim = (None,) + param.value.shape[1:]
        shape = tf.TensorShape(new_dim)
        var = tf.Variable(param.value, shape=shape)
        return var
    return tf.Variable(param.value)


def convert_to_tensor(param):
    if param.indices is not None:
        return tf.IndexedSlices(param.value, param.indices)
    return tf.convert_to_tensor(param.value)


class PServer(object):
    def __init__(self, opt, grads_to_wait):
        self.kvstore = KVStoreServicer()
        self.opt = opt
        self.grads_to_wait = grads_to_wait
        self.use_async = False
        self.version = -1
        if self.grads_to_wait == 1:
            self.use_async = True

        self.grads_num = {}

    def get_param(self, grad):
        if len(grad.indices) == 0:
            return self.kvstore.get_param(grad.name)
        else:
            y, idx = tf.unique(grad.indices)
            grad.indices = idx.numpy()
            param = self.kvstore.get_embedding_param(grad.name, y.numpy())
            return param

    def set_param(self, param):
        if len(param.indices) == 0:
            self.kvstore.set_param(param.name, param)
        else:
            self.kvstore.set_embedding_param(
                param.name, param.indices, param.value
            )

    def get_gradient(self):
        if self.use_async:
            cur_gradient = self.kvstore.grads.get()
            return cur_gradient

        cur_gradient = self.kvstore.grads.get()
        while (cur_gradient != self.version) and (
            self.grads_num[cur_gradient.name] < self.grads_to_wait
        ):
            cur_gradient = self.kvstore.grads.get()

        return cur_gradient

    def apply_gradient(self):
        grad = self.get_gradient()
        param = self.get_param(grad)

        param_var = convert_to_var(param)

        grads_and_params = list(zip([convert_to_tensor(grad)], [param_var]))

        self.opt.apply_gradients(grads_and_params)
        print("apply_gradient sucess")

        param.value = param_var.numpy()
        self.set_param(param)

    def start(self, endpoint):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        core_pb2_grpc.add_KVStoreServicer_to_server(self.kvstore, server)
        print("Starting server. Listening on endpoint %s." % endpoint)
        server.add_insecure_port(endpoint)
        server.start()
        try:
            while True:
                self.apply_gradient()
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint", type=str)
    args = parser.parse_args()

    opt = SGD(lr=0.01)
    ps = PServer(opt, 1)
    ps.start(args.endpoint)
