from kvstore_service import KVStoreServicer
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

import grpc
from concurrent import futures
import time

import kvstore_pb2
import kvstore_pb2_grpc

class PServer(object):
    def __init__(self, opt):
        self.kvstore = KVStoreServicer()
        self.opt = opt
    
    def apply_gradient(self):
        cur_gradient = self.kvstore.grads.get()
        grad_name, grad = cur_gradient[0], cur_gradient[1]
        param_name = grad_name[0:-5]
        param = self.kvstore.query_db(param_name)

        grad = tf.convert_to_tensor(grad)
        param_var = tf.Variable(tf.convert_to_tensor(param))

        grads_and_params = list(zip([grad], [param_var]))
        self.opt.apply_gradients(grads_and_params)
        print('apply_gradient sucess')

        self.kvstore.update_db((param_name, param_var.numpy()))

    def start(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        kvstore_pb2_grpc.add_KVStoreServicer_to_server(
            self.kvstore, server)
        print('Starting server. Listening on port 50051.')
        server.add_insecure_port('[::]:50051')
        server.start()
        try:
            while True:
                self.apply_gradient()
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == "__main__":
    opt = SGD(lr=0.1)
    ps = PServer(opt)
    ps.start()