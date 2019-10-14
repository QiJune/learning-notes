import queue

from google.protobuf import empty_pb2
from ps_mini.core_pb2_grpc import GradientQueueServicer
from ps_mini.tensor import Tensor, deserialize_from_pb


class GradientQueue(GradientQueueServicer):
    def __init__(self):
        self.grad_queue = queue.Queue()

    def push_grad(self, request, _):
        tensor = Tensor()
        deserialize_from_pb(request, tensor)
        self.grad_queue.put(tensor)
        return empty_pb2.Empty()

    def push_embedding_grad(self, request, _):
        tensor = Tensor()
        deserialize_from_pb(request, tensor)
        self.grad_queue.put(tensor)
        return empty_pb2.Empty()

    def get_grad(self):
        return self.grad_queue.get()
