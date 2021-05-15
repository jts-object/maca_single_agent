import tensorflow as tf
import numpy as np
# from transformer import Transformer


class Net(object):
    def __init__(self, hidden_size=[16, 16, 8, 1]):
        self.batch_size = 10
        self.seq_max_len = 5
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.seq_max_len, 4], name='input_data')
        self.real_out = tf.placeholder(dtype=tf.float32, shape=[self.batch_size], name='input_data')

        self.hidden_size = hidden_size
        self.build_net()
        self.build_loss()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        self.transformer = Transformer(
            _raw_input=self.input_data, num_heads=8, d_model=16, d_ff=64, num_enc_layers=6, num_dec_layers=6, 
            drop_rate=0.1, warmup_steps=400, pos_encoding_type='sinusoid', ls_epsilon=0.1, use_label_smoothing=False, 
            model_name='transformer', batch_size=self.batch_size, seq_len=self.seq_max_len, 
            tf_sess_config=None, kwargs=None)
        self.transformer.build_model()
        enc_out = self.transformer.enc_out
        self.enc_out = enc_out
        hid_state = tf.concat(tf.unstack(enc_out, axis=1), axis=-1)

        hid_state = tf.layers.dense(inputs=hid_state, units=64, activation=None)

        for size in self.hidden_size:
            hid_state = tf.layers.dense(inputs=hid_state, units=size, activation=tf.nn.relu)
        
        self.net_out = tf.squeeze(hid_state, axis=-1)

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.net_out - self.real_out))
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 

    def train(self, input_data, real_out):
        feed_dict = {self.input_data: input_data, self.real_out: real_out}
        loss, _, net_out, enc_out, var_list = self.sess.run([self.loss, self.train_op, self.net_out, self.enc_out, self.var_list], feed_dict=feed_dict)

        print('net_out:', net_out)
        print('loss:', loss)
        # print('encoder out:', enc_out[0])
        print('var list:', var_list[0])
        print('var list:', var_list[5])

# activation = tf.nn.relu
# img_plh = tf.placeholder(tf.float32, [None, 3, 3, 3])
# label_plh = tf.placeholder(tf.float32, [None])
# layer = img_plh
# buffer = []
# ks_list = list(range(1, 10, 1)) + list(range(9, 0, -1))
# for ks in ks_list:
#     buffer.append(tf.layers.conv2d(layer, 9, ks, 1, "same", activation=activation))
# layer = tf.concat(buffer, 3)
# layer = tf.layers.conv2d(layer, 1, 3, 1, "valid", activation=activation)
# layer = tf.squeeze(layer, [1, 2, 3])
# loss_op = tf.reduce_mean(tf.abs(label_plh - layer))
# optimizer = tf.train.AdamOptimizer()
# train_op = optimizer.minimize(loss_op)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#     for i in range(4):
#         _, loss, v_list = sess.run([train_op, loss_op, var_list], {img_plh: np.random.normal(size=[2, 3, 3, 3]), label_plh: np.random.normal(size=[2])})

#         print('loss:', loss)
#         print('var list:', v_list[0])
#         print('var list:', v_list[6])
#         print('var list:', v_list[10])
        
from tensorflow.distributions import Categorical

input_ph = tf.placeholder(shape=[5, 4], dtype=tf.float32, name='input')
act_mask_ph = tf.placeholder(shape=[5, 2], dtype=tf.float32, name='mask')
real_out = tf.placeholder(shape=[5], dtype=tf.float32, name='real_out')

out_head1_logits = tf.layers.dense(inputs=input_ph, units=10, activation=tf.nn.relu)
out_head1_logits = tf.layers.dense(inputs=out_head1_logits, units=2, activation=tf.nn.relu)
out_head1_logits = out_head1_logits + act_mask_ph
# out_head1_logits = tf.reduce_mean(out_head1_logits, axis=-1)
out_head1_prob = -tf.reduce_max(out_head1_logits, axis=-1)


out_head1_dist = Categorical(logits=out_head1_logits)
out_head1_action = out_head1_dist.sample()
# out_head1_prob = out_head1_dist.prob(out_head1_action)

loss = tf.reduce_sum(tf.square(real_out - out_head1_prob))
optimizer = tf.train.AdamOptimizer(1e-3)
train_op = optimizer.minimize(loss)
gvs = optimizer.compute_gradients(loss)
grads = [grad for grad, _ in gvs if grad is not None]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print('var list:', var_list)

    for i in range(4):
        input_val = np.random.normal(size=[5, 4])
        real_out_val = np.random.normal(size=[5])
        act_mask_val = np.array([
            [0.,    -1.e6],
            [0.,    0.],
            [-1.e6, 0.],
            [0.,    -1.e6],
            [-1.e6, 0],])
        print('real_out_val', real_out_val)

        _, loss_res, v_list_res, grads_res = sess.run([train_op, loss, var_list, grads], {
            input_ph: input_val, 
            act_mask_ph: act_mask_val, 
            real_out: real_out_val,})

        print('loss:', loss_res)
        print('grads_res:', grads_res)


# if __name__ == '__main__':
#     net = Net()

#     for i in range(5):
#         input_data = np.ones(shape=[10, 5, 4])
#         real_out = np.random.normal(size=[10])
    
#         net.train(input_data, real_out)
