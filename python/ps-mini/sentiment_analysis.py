import os

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
assert tf.__version__.startswith("2.")

tf.random.set_seed(22)
np.random.seed(22)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# truncate and pad input sequences
max_review_length = 80
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=top_words
)

print("Pad sequences (samples x time)")
x_train = tf.keras.preprocessing.sequence.pad_sequences(
    X_train, maxlen=max_review_length
)
x_test = tf.keras.preprocessing.sequence.pad_sequences(
    X_test, maxlen=max_review_length
)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)


class RNN(tf.keras.Model):
    def __init__(self, units):
        super(RNN, self).__init__()

        self.rnn = tf.keras.layers.LSTM(units)
        self.embedding = tf.keras.layers.Embedding(
            top_words, 100, input_length=max_review_length
        )
        self.fc = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training):
        x = self.embedding(inputs)
        x = self.rnn(x)
        x = self.fc(x)
        return x


def main():
    units = 64
    batch_size = 32
    epochs = 1
    model = RNN(units)

    optimizer = tf.keras.optimizers.Adam(0.001)
    acc_meter = tf.keras.metrics.Accuracy()
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    train_data = train_data.batch(batch_size).repeat(epochs)
    for step, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:
            outputs = model.call(x, training=True)
            outputs = tf.reshape(outputs, [-1])
            loss = tf.keras.losses.binary_crossentropy(
                y, outputs, from_logits=False
            )

        acc_meter.update_state(
            tf.where(outputs < 0.5, x=tf.zeros_like(y), y=tf.ones_like(y)), y
        )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(
                step, "loss:", float(loss), "acc:", acc_meter.result().numpy()
            )
            acc_meter.reset_states()


if __name__ == "__main__":
    main()
