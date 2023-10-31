import numbers
import random

import tensorflow as tf
import numpy as np


class GroupBasedDropOut(tf.keras.layers.Layer):
    def __init__(self, group_size, **kwargs):
        super(GroupBasedDropOut, self).__init__(**kwargs)
        self.mask_shape = None
        self.group_size = group_size
        if group_size != 0:
            self.scalar = 1 / ((group_size - 1) / group_size)

    def build(self, input_shape):
        if self.group_size != 0 and input_shape.as_list()[1] % self.group_size != 0:
            raise Exception("Input shape is not a multiple of level_size")
        self.mask_shape = input_shape

    def call(self, inputs, *args, **kwargs):
        if (isinstance(self.group_size, numbers.Real) and self.group_size == 0) or not kwargs['training']:
            return tf.identity(inputs)

        return tf.math.multiply(tf.identity(inputs), self.create_mask(inputs[0]))

    def create_mask(self, input_tensor):
        arr = np.full(input_tensor.get_shape().as_list()[0], self.scalar, dtype='float32')
        for i in range(0, len(arr), self.group_size):
            temp = random.randint(0, self.group_size - 1)
            arr[i + temp] = 0

        arr = arr[None, :]
        return tf.convert_to_tensor(arr)
