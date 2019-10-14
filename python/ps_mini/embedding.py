import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


class Embedding(tf.keras.layers.Layer):
    def __init__(self,
                 output_dim,
                 embedding_initializer="uniform",
                 input_length=None,
                 **kwargs):
        if "input_shape" not in kwargs and input_length:
            kwargs["input_shape"] = (input_length, )
        super(Embedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.embedding_initializer = embedding_initializer
        self.input_length = input_length

        self.tape = None
        self.kvstore_client = None

        @tf_utils.shape_type_conversion
        def compute_output_shape(self, input_shape):
            # this function is taken from
            # tf.keras.layers.Embedding.compute_output_shape
            # https://github.com/tensorflow/tensorflow/blob/3f3c728bf80e0fd6653744318cbbfe1454c6ddca/tensorflow/python/keras/layers/embeddings.py#L156
            if self.input_length is None:
                return input_shape + (self.output_dim, )
            else:
                if isinstance(self.input_length, (list, tuple)):
                    in_lens = list(self.input_length)
                else:
                    in_lens = [self.input_length]
                if len(in_lens) != len(input_shape) - 1:
                    raise ValueError(
                        '"input_length" is %s, '
                        "but received input has shape %s" %
                        (str(self.input_length), str(input_shape)))
                else:
                    for i, (s1, s2) in enumerate(zip(in_lens,
                                                     input_shape[1:])):
                        if s1 is not None and s2 is not None and s1 != s2:
                            raise ValueError(
                                '"input_length" is %s, '
                                "but received input has shape %s" %
                                (str(self.input_length), str(input_shape)))
                        elif s1 is None:
                            in_lens[i] = s2
            return (input_shape[0], ) + tuple(in_lens) + (self.output_dim, )

    @property
    def name(self):
        return self._name

    def set_tape(self, tape):
        self.tape = tape

    def set_kvstore_client(self, kvstore_client):
        self.kvstore_client = kvstore_client

    def call(self, input):
        ids = tf.convert_to_tensor(input, name="embedding_ids")
        flat_ids = tf.reshape(ids, [-1])
        unique_ids, idx = tf.unique(flat_ids)

        self.batch_embedding = self.kvstore_client.pull_embedding_param(
            self.name,
            unique_ids.numpy(),
            self.output_dim,
            self.embedding_initializer,
        )
        self.batch_embedding_tensor = tf.convert_to_tensor(
            self.batch_embedding.value)
        if self.tape:
            self.tape.watch(self.batch_embedding_tensor)

        outputs = tf.gather(self.batch_embedding_tensor, idx)
        # tf.reshape does not support shape with None. Replace None with -1.
        if ids.get_shape().rank == 2:
            output_shape = (-1, ids.get_shape()[1], self.output_dim)
        else:
            output_shape = ids.get_shape().concatenate(self.output_dim)
        outputs = tf.reshape(outputs, output_shape)
        return outputs

    def reset(self):
        self.tape = None
