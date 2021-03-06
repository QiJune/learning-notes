import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time

import mnist
import tensorflow as tf
from tensorflow.python.eager import context, tape


def random_batch(batch_size):
    """Create synthetic resnet50 images and labels for testing."""
    shape = (28, 28)
    shape = (batch_size, ) + shape

    num_classes = 10
    images = tf.random.uniform(shape)
    labels = tf.random.uniform([batch_size],
                               minval=0,
                               maxval=num_classes,
                               dtype=tf.int32)

    return images, labels


def compute_gradients(model, images, labels, num_replicas=1):
    with tf.GradientTape() as grad_tape:
        logits = model(images, training=True)
        labels = tf.reshape(labels, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=labels)
        if num_replicas != 1:
            loss /= num_replicas

    # TODO(b/110991947): We can mistakenly trace the gradient call in
    # multi-threaded environment. Explicitly disable recording until
    # this is fixed.
    with tape.stop_recording():
        grads = grad_tape.gradient(loss, model.trainable_variables)
    return grads


def apply_gradients(model, optimizer, gradients):
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


class ModelTest(tf.test.TestCase):
    def _test_train(self, execution_mode=None):
        start = time.process_time()
        model = mnist.custom_model()

        with tf.device("CPU"), context.execution_mode(execution_mode):
            optimizer = tf.keras.optimizers.SGD(0.1)
            images, labels = random_batch(1000)
            apply_gradients(model, optimizer,
                            compute_gradients(model, images, labels))
            context.async_wait()
        end = time.process_time()
        print("time: ", end - start)

    def test_train(self):
        self._test_train()

    def test_train_async(self):
        self._test_train(execution_mode=context.ASYNC)


if __name__ == "__main__":
    tf.test.main()
