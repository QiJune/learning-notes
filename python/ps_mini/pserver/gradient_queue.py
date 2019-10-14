import queue


class GradientQueue(object):
    def __init__(self):
        self.grad_queue = queue.Queue()

    def put_grad(self, grad):
        self.grad_queue.put(grad)

    def get_grad(self):
        return self.grad_queue.get()
