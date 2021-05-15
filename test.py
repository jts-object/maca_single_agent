import tensorflow as tf
import numpy as np 
from tensorflow.contrib.rnn import GRUCell 

cell = GRUCell(num_units=32, activation=tf.nn.relu)
input_tensor = tf.random.normal(shape=[3, 5])
init_state = tf.zeros(shape=[3, 32])

# 只经过经过一次cell的运算
output, state = cell(input_tensor, init_state)
print('shape of output and state', output.shape.as_list(), state.shape.as_list())


# output, state = tf.nn.dynamic_rnn(
#     cell=cell, inputs=input_tensor, dtype=tf.float32, initial_state=tf.random.normal(shape=[10, 32]))

# print('shape of :', output.shape.as_list(), state.shape.as_list())