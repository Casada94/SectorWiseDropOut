import numbers
import random

import tensorflow as tf
import numpy as np


class MultiLevel(tf.keras.layers.Layer):
    def __init__(self, level_size, use_all_nodes, **kwargs):
        super(MultiLevel, self).__init__(**kwargs)
        self.offsets = None
        self.num_masked = None
        self.mask = None
        self.use_all_nodes = use_all_nodes
        self.level_size = level_size
        self.pseudo_logits = tf.ones(shape=(1, level_size))

    def build(self, input_shape):
        if self.level_size != 0 and input_shape.as_list()[1] % self.level_size != 0:
            raise Exception("Input shape is not a multiple of level_size")
        self.num_masked = int(input_shape.as_list()[1]) // self.level_size
        self.mask = tf.zeros((1, self.num_masked))
        self.offsets = tf.expand_dims(
            tf.range(self.num_masked) * self.level_size, axis=0)

    def call(self, inputs, *args, **kwargs):
        tf.print(inputs.get_shape())
        if isinstance(self.level_size, numbers.Real) and self.level_size == 0:
            return tf.identity(inputs)

        slice_ids = tf.random.categorical(
            self.pseudo_logits, self.num_masked, dtype=tf.int32)
        slice_ids = tf.reshape(slice_ids + self.offsets, (1, self.num_masked))
        masked_tensor = tf.tensor_scatter_nd_update(inputs, slice_ids, self.mask)

        if self.use_all_nodes:
            return tf.math.multiply(masked_tensor, self.level_size)
        else:
            return masked_tensor
        # return tf.math.multiply(tf.identity(inputs), self.create_mask(inputs[0]))

    # @tf.function
    # def create_mask(self, input_tensor):
    #     # self.seed += 1
    #     # self.rand.seed(self.seed)
    #     arr = np.full(input_tensor.get_shape().as_list()[0], 0, dtype='float32')
    #     for i in range(0, len(arr), self.level_size):
    #         # temp = self.rand.randint(0, self.level_size - 1)
    #         temp = tf.random.uniform(shape=(), minval=0, maxval=self.level_size, dtype=tf.int32)
    #         # arr[i + temp] = self.scalar if not self.use_all_nodes else 1
    #         arr[i + int(temp)] = self.scalar if not self.use_all_nodes else 1
    #     tf.print(arr)
    #     arr = arr[None, :]
    #     return tf.convert_to_tensor(arr)
