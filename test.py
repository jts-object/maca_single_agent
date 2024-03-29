import tensorflow as tf
import numpy as np 
from tensorflow.contrib.rnn import GRUCell 
import tensorflow.contrib as tc

# cell = GRUCell(num_units=32, activation=tf.nn.relu)
# input_tensor = tf.random.normal(shape=[3, 5])
# init_state = tf.zeros(shape=[3, 32])

# # 只经过经过一次cell的运算
# output, state = cell(input_tensor, init_state)
# print('shape of output and state', output.shape.as_list(), state.shape.as_list())

# output, state = tf.nn.dynamic_rnn(
#     cell=cell, inputs=input_tensor, dtype=tf.float32, initial_state=tf.random.normal(shape=[10, 32]))

# print('shape of :', output.shape.as_list(), state.shape.as_list())


# head0_pointer_prob = tf.placeholder(dtype=tf.float32, shape=[2, 3, 4], name='prob')
# act_prob_head0_ph = tf.placeholder(dtype=tf.float32, shape=[2, 3, 4], name='prob')
# head0_pointers = tf.placeholder(dtype=tf.int32, shape=[2, 3], name='pointer')

# range_ = tf.range(head0_pointer_prob.shape.as_list()[1])
# # range_ = tf.zeros(head0_pointer_prob.shape.as_list()[1], dtype=tf.int32)
# range_ = tf.tile(tf.expand_dims(range_, axis=0), [head0_pointer_prob.shape.as_list()[0], 1])
# index = tf.stack([range_, head0_pointers], axis=0)

# # range_ = tf.range(head0_pointer_prob.shape.as_list()[0])
# range_ = tf.zeros(head0_pointer_prob.shape.as_list()[0], dtype=tf.int32)
# range_ = tf.tile(tf.expand_dims(range_, axis=1), [1, head0_pointer_prob.shape.as_list()[1]])
# res = tf.concat([tf.expand_dims(range_, axis=0), index], axis=0)
# final_index = tf.transpose(res, [1, 2, 0])

# head0_prob = tf.gather_nd(act_prob_head0_ph, final_index)     # self.act_prob_head0 存储的是旧策略对应的概率
# # head0_prob = tf.clip_by_value(head0_prob, 1e-3, 1e5)
# old_neglogp_head0 = tf.clip_by_value(-tf.log(tf.reduce_prod(head0_prob, axis=1)), -1.e3, 1.e3)     # 维度: [batch]

# with tf.Session() as sess:
#     head0_pointer_prob_val = np.random.normal(size=[2, 3, 4])
#     act_prob_head0_val = np.random.normal(size=[2, 3, 4])
#     head0_pointer_val = np.array([[0, 1, 2], [1, 2, 3]])
    
#     feed_dict = {
#         head0_pointer_prob: head0_pointer_prob_val,
#         act_prob_head0_ph: act_prob_head0_val, 
#         head0_pointers: head0_pointer_val,
#     }
#     # print('head0_pointer_prob_val:', head0_pointer_prob_val)
#     # print('act_prob_head0_val:', act_prob_head0_val)
#     # print('head0_pointer_val:', head0_pointer_val)

#     res1, res2 = sess.run([head0_prob, final_index], feed_dict=feed_dict)


# inp = tf.placeholder(shape=[3, 32], name='input', dtype=tf.float32)
# out_norm = tc.layers.layer_norm(inp, center=True, scale=True)
# # out = tf.layers.dense(inputs=inp, units=1, activation=None, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
# # out = tf.layers.dense(inputs=inp, units=1, activation=None)
# var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     inp_val = np.random.normal(size=[3, 32])
#     res1, varss  = sess.run([out_norm, var_list], feed_dict={inp: inp_val})
#     # print('inp_val:', inp_val)
#     print('res1:', res1)
    
#     varss = np.array(varss[0])
#     print(varss.mean(), varss.var())
#     print(varss.T)

x = np.array([   0., 0., 0.,  0., 0., 45.469532, 0.,       509.19467,   349.08994 ,    0. ,       224.57466 ,  195.98547,
    0.,          0.,          0. ,       413.90326,   108.0392,      0.,
    0. ,       704.9939 ,     0.  ,        0.    ,      0.      ,  122.696396,
  700.40704 ,    0.  ,        0.  ,     1146.125,       0.,          0.,
    0. ,         0.      ])

w = np.array([[-0.24952722],
 [ 0.7671394 ],
 [-0.30935612],
 [ 1.6671817 ],
 [ 0.21854721],
 [-1.3078921 ],
 [-0.54096204],
 [-1.1892422 ],
 [-0.9158353 ],
 [-1.082895  ],
 [-0.08779025],
 [-0.40155125],
 [ 0.10024366],
 [-0.05633254],
 [-0.4799334 ],
 [-0.9700537 ],
 [ 0.0267899 ],
 [ 0.06524917],
 [ 1.8132304 ],
 [-0.47081098],
 [-0.9597537 ],
 [ 1.0136133 ],
 [-1.2962543 ],
 [ 0.04875791],
 [-1.1115236 ],
 [ 0.3849017 ],
 [ 0.64772236],
 [-1.6753253 ],
 [-0.01278995],
 [ 0.46187627],
 [-0.10455292],
 [ 0.9743693 ]])

print(w.shape, x.shape)

print(w.T @ x)
