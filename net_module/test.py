import tensorflow as tf 
from tensorflow.python import nest
from collections import namedtuple 
import numpy as np 


# 测试 tf.name_scope 的影响范围===================================================
def create_ph(name_):
    return tf.placeholder(dtype=tf.float32, name=name_, shape=[2, 3])

with tf.variable_scope('test'):
    a_ph = create_ph(name_='a')
    
print(a_ph)


# 测试 while_loop 方法===================================================
# def func(i):
#     i = i + 1
#     print('i am in func')
#     return i

# def cond(i, n):
#     return i < n

# def body(i, n):
#     i = func(i)
#     print('i am in body')
#     return i, n

# i = tf.get_variable("i_value", dtype=tf.int32, shape=[], initializer=tf.ones_initializer())
# n = tf.constant(10)

# import collections
# class test(collections.namedtuple("state",
#                            ("cell_state", "time"))):
#     def __init__(self, cell_state, time):
#         super(test, self).__init__()

# t1 = test(1, 2)
# history = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
# num_list_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=[5, 5])
# num_list_array_stack = num_list_array.stack()
# num = 0
# num_list = tf.placeholder(dtype=tf.int32, shape=[5, 5], name='num_list')
# mask_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, element_shape=[1, 5])


# def cond(history, num_list, num, num_list_array, mask_array):
#     return num < 5

# def body(history, num_list, num, num_list_array, mask_array):
    
#     # history = history.write(num, tf.constant(0))
#     history_tensor = tf.cast(history.stack(), tf.int32)
#     # print('shape of history tensor', history_tensor.shape.as_list())
#     history_mask = tf.one_hot(indices=history_tensor, depth=5, dtype=tf.int32)
#     # print('shape of history mask', history_mask.shape.as_list())
#     history_mask = 1 - tf.reduce_sum(history_mask, axis=0, keepdims=True)
#     mask_array = mask_array.write(num, history_mask)

#     history = history.write(num, num)

#     history_mask = tf.tile(history_mask, [5, 1])
#     rest_mask = tf.ones(shape=[4 - num, 5], dtype=tf.int32)
#     # history_mask = tf.concat([history_mask, rest_mask], axis=0)
    
#     num_list_ = num_list * history_mask
    
#     num_list_array = num_list_array.write(num, num_list_)

#     num += 1

#     return history, num_list, num, num_list_array, mask_array


# history_, num_list_, num_, num_list_array_, mask_array_ = tf.while_loop(
#     cond, body, 
#     loop_vars=(history, num_list, num, num_list_array, mask_array), maximum_iterations=5,)      # shape_invariants=(tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([])))

# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     num_list_val = np.random.randint(low=1, high=10, size=[5, 5])
#     res1, res2, res3, res4 = sess.run([history_.stack(), num_list_, mask_array_.stack(), num_list_array_.stack()], feed_dict={num_list: num_list_val})
#     print('res4', res4)

    # a = tf.constant([[1, 2, 3],[2, 3, 4]]) 
    # mask = tf.one_hot(indices=a, depth=5)
    # mask = tf.reduce_sum(mask, axis=0)
    # print('mask shape', mask.shape.as_list())
    # res = sess.run(mask)
    # print('res:', res4)
    # print('num_list_val', num_list_val)


# 测试 wrapperState ====================================================
# class State(namedtuple('wrapperState', ('state', 'history'))):
#     def clone(self):
#         pass 
# state = State(1, 2)
# a = {'state': 3}
# state._replace(**a)
# print(state.state)

# 测试cond函数 ============================================================

# init_time_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
# init_time = tf.constant(0) 
# print('init_time', init_time)

# def cond(time, time_ta):
#     return True

# def body(time, time_ta):
#     res = tf.cond(time > tf.constant(5), lambda: -1, lambda: time)
#     time_ta = time_ta.write(time, res)

#     return time + 1, time_ta

# final_time, final_time_ta = tf.while_loop(cond, body, loop_vars=(init_time, init_time_ta), maximum_iterations=10)

# with tf.Session() as sess:
#     res = sess.run(final_time_ta.stack())
#     print(res)


# 测试 nest 的相关方法=====================================================
# a = tf.placeholder(dtype=tf.float32, shape=[2, 3, 4], name='a')
# b = a.get_shape()[1:].is_fully_defined()

# def check_dim(tensor):
#     print('yes')

# b = [[a], a, a]
# nest.map_structure(check_dim, b)

# c = nest.flatten(b)
# c = tf.sequence_mask([5, 5, 5], 5, dtype=tf.float32)
# flag = tf.assert_positive(c, message='False')


# 测试 zero_state ======================================================
# from tensorflow.python.ops import rnn_cell_impl
# _zero_state_tensors = rnn_cell_impl._zero_state_tensors
# def create_zero_outputs(size, dtype, batch_size):
#     """Create a zero outputs Tensor structure."""
#     def _create(s, d):
#         return _zero_state_tensors(s, batch_size, d)
    
#     return nest.map_structure(_create, size, dtype)

# res = create_zero_outputs([2, 3], [tf.float32, tf.int32], 3)

# var1 = tf.constant([[1, 2], [3, 4]])
# var2 = tf.constant([[1], [0]])
# res = var1 * var2


# 测试生成 mask 的方法===================================================
# MAXLEN = 5
# EMBEDDING = 4
# SEQ_LEN = 5

# index_list = [tf.placeholder(dtype=tf.int32, shape=[2], name='{}_pointer'.format(i)) for i in range(2)]

# index_onehot_list = [tf.one_hot(indices=index, depth=SEQ_LEN, on_value=0, off_value=1) for index in index_list]
# index_onehot_list = [tf.tile(tf.expand_dims(index, axis=1), [1, EMBEDDING]) for index in index_onehot_list]
# index_onehot_list = [tf.cast(index, tf.bool) for index in index_onehot_list]
# index_onehot_list = [tf.tile(tf.expand_dims(tf.reduce_all(index, axis=0), axis=1), [1, EMBEDDING]) for index in index_onehot_list]
# index_onehot = tf.stack(index_onehot_list, axis=0)
# index_onehot = tf.cast(index_onehot, tf.float32)
# print(index_onehot.shape.as_list())
# index_embed = tf.tile(tf.expand_dims(index_embed, axis=1), [1, 2])


# sess开启=============================================================
#sess = tf.Session()
#c = sess.run(index_onehot, feed_dict={index: 2 for index in index_list})
#print(c)

