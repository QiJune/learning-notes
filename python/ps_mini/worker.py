import argparse
import os

import numpy as np
import tensorflow as tf
from embedding import Embedding
from kvstore_client import KVStoreClient
from tensor import Tensor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
assert tf.__version__.startswith("2.")

tf.random.set_seed(22)
np.random.seed(22)


class RNN(tf.keras.Model):
    def __init__(self, units, max_review_length):
        super(RNN, self).__init__()

        self.rnn = tf.keras.layers.LSTM(units)
        self.embedding = Embedding(100, input_length=max_review_length)
        self.fc = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training):
        x = self.embedding(inputs)
        x = self.rnn(x)
        x = self.fc(x)
        return x


def get_dataset(max_review_length):
    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000

    (X_train,
     y_train), (X_test,
                y_test) = tf.keras.datasets.imdb.load_data(num_words=top_words)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(
        X_train, maxlen=max_review_length)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(
        X_test, maxlen=max_review_length)

    return x_train, y_train, x_test, y_test


class Worker(object):
    def __init__(self, master_endpoint, worker_id):
        self.kvstore_client = KVStoreClient(master_endpoint, worker_id)
        self.metric = tf.keras.metrics.Accuracy()
        self.init = False
        self.model = RNN(64, 80)
        self.x_train, self.y_train, self.x_test, self.y_test = get_dataset(80)
        self.embedding_layer = []
        self.embedding_layer_info = []
        self._init_embedding_layer()

    def _init_embedding_layer(self):
        for layer in self.model.layers:
            if isinstance(layer, Embedding):
                self.embedding_layer.append(layer)
                layer.set_kvstore_client(self.kvstore_client)
                name = layer.name
                output_dim = layer.output_dim
                initializer = layer.embedding_initializer
                self.embedding_layer_info.append(
                    (name, output_dim, initializer))

    def forward_backward(self, x, y):
        with tf.GradientTape() as tape:
            for layer in self.embedding_layer:
                layer.set_tape(tape)
            outputs = self.model.call(x, training=True)
            if not self.init:
                for var in self.model.trainable_variables:
                    param = Tensor(name=var.name, value=var.numpy())
                    self.kvstore_client.push_param(param)

                for embedding_info in self.embedding_layer_info:
                    name, output_dim, initializer = embedding_info
                    param = Tensor(name=name,
                                   value=np.ones(shape=output_dim),
                                   initializer=initializer)
                    self.kvstore_client.push_embedding_param(param)

                self.init = True
            outputs = tf.reshape(outputs, [-1])
            self.loss = tf.keras.losses.binary_crossentropy(y,
                                                            outputs,
                                                            from_logits=False)
            self.metric.update_state(
                tf.where(outputs < 0.5, x=tf.zeros_like(y), y=tf.ones_like(y)),
                y,
            )
            vars = []
            vars.extend(self.model.trainable_variables)
            for layer in self.embedding_layer:
                vars.append(layer.batch_embedding_tensor)
            self.grads = tape.gradient(self.loss, vars)
            print(len(self.grads))
            print(len(self.model.trainable_variables))
            exit(0)

    def train(self):
        train_data = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train))
        train_data = train_data.batch(8).repeat(2)

        for step, (x, y) in enumerate(train_data):
            self.forward_backward(x, y)
            if step % 100 == 0:
                print(
                    step,
                    "loss:",
                    float(self.loss),
                    "acc:",
                    self.metric.result().numpy(),
                )
                self.metric.reset_states()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pserver_endpoints', nargs='+', required=True)
    parser.add_argument("-i", "--worker_id", type=int)
    args = parser.parse_args()

    worker = Worker(args.pserver_endpoints, args.worker_id)
    worker.train()
