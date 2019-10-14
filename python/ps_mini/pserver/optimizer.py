import tensorflow as tf


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


class Optimizer(object):
    def __init__(self, opt, grad_queue, kvstore):
        self.opt = opt
        self.grad_queue = grad_queue
        self.kvstore = kvstore

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

    def get_grad(self):
        return self.grad_queue.get_grad()

    def apply_gradients(self):
        grad = self.get_grad()
        param = self.get_param(grad)

        param_var = convert_to_var(param)

        grads_and_params = list(zip([convert_to_tensor(grad)], [param_var]))

        self.opt.apply_gradients(grads_and_params)
        print("apply_gradient sucess")

        param.value = param_var.numpy()
        self.set_param(param)
