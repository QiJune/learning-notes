import time

import mnist
import tensorflow as tf
import util
from tensorflow.python.eager import context, tape


def compute_gradients(model, images, labels, num_replicas=1):
    with tf.GradientTape() as grad_tape:
        logits = model(images, training=True)
        labels = tf.reshape(labels, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
        tf.compat.v2.summary.write("loss", loss)
        if num_replicas != 1:
            loss /= num_replicas

    # TODO(b/110991947): We can mistakenly trace the gradient call in
    # multi-threaded environment. Explicitly disable recording until
    # this is fixed.
    with tape.stop_recording():
        grads = grad_tape.gradient(loss, model.variables)
    return grads


def apply_gradients(model, optimizer, gradients):
    optimizer.apply_gradients(zip(gradients, model.variables))


class ModelTest(tf.test.TestCase):
    def _test_train(self, execution_mode=None):
        start = time.process_time()
        model = mnist.custom_model()

        with tf.device("CPU"), context.execution_mode(execution_mode):
            optimizer = tf.keras.optimizers.SGD(0.1)
            images, labels = util.random_batch(1000)
            apply_gradients(
                model, optimizer, compute_gradients(model, images, labels)
            )
            context.async_wait()
        end = time.process_time()
        print("time: ", end - start)

    def test_train(self):
        self._test_train()

    def test_train_async(self):
        self._test_train(execution_mode=context.ASYNC)


if __name__ == "__main__":
    tf.test.main()
