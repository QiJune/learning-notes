import tensorflow as tf


def random_batch(batch_size):
    """Create synthetic resnet50 images and labels for testing."""
    shape = (28, 28)
    shape = (batch_size,) + shape

    num_classes = 10
    images = tf.random.uniform(shape)
    labels = tf.random.uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.int32
    )

    return images, labels
