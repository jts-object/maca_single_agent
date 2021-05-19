import tensorflow as tf 
import numpy as np 
from .ptrnet import PointerNet
from .transformer import Transformer
from tensorflow.contrib.rnn import GRUCell
from tensorflow.distributions import Categorical
import tensorflow.contrib as tc


class Network(object):
    def __init__(
        self, _raw_input1, _raw_input2, memory_mask_1, memory_mask_2, action_mask_1, 
        num_heads=8, d_model=32, d_ff=128, num_enc_layers=6, num_dec_layers=6,
        drop_rate=0.1, warmup_steps=400, pos_encoding_type='sinusoid',
        ls_epsilon=0.1, use_label_smoothing=False, model_name='model',
        tf_sess_config=None, n_pointers=3, batch_size=10, seq_max_len=8,
        learning_rate=0.001, cell=tf.contrib.rnn.GRUCell, n_layers=6, n_units=50, **kwargs):
        
        # raw_obs_1, raw_obs_2 = _raw_input1['obs'], _raw_input2['obs']
        self.transformer1 = Transformer(
            _raw_input=_raw_input1, num_heads=num_heads, d_model=d_model, d_ff=d_ff, num_enc_layers=num_enc_layers, 
            num_dec_layers=num_dec_layers, drop_rate=drop_rate, warmup_steps=warmup_steps, 
            pos_encoding_type=pos_encoding_type, ls_epsilon=ls_epsilon, 
            use_label_smoothing=use_label_smoothing, model_name=model_name + '_transformer_1', batch_size=batch_size, seq_len=seq_max_len, 
            tf_sess_config=None, kwargs=kwargs)
        self.transformer2 = Transformer(
            _raw_input=_raw_input2, num_heads=num_heads, d_model=d_model, d_ff=d_ff, num_enc_layers=num_enc_layers, 
            num_dec_layers=num_dec_layers, drop_rate=drop_rate, warmup_steps=warmup_steps, 
            pos_encoding_type=pos_encoding_type, ls_epsilon=ls_epsilon, 
            use_label_smoothing=use_label_smoothing, model_name=model_name + '_transformer_2', batch_size=batch_size, seq_len=seq_max_len, 
            tf_sess_config=None, kwargs=kwargs)

        self.ptr_net1 = PointerNet(
            n_pointers=n_pointers, batch_size=batch_size, seq_max_len=seq_max_len, 
            learning_rate=0.001, cell=tf.contrib.rnn.GRUCell, n_layers=n_layers, n_units=n_units, name=model_name + '_ptr_net_1')
        self.ptr_net2 = PointerNet(
            n_pointers=1, batch_size=batch_size, seq_max_len=seq_max_len, 
            learning_rate=0.001, cell=tf.contrib.rnn.GRUCell, n_layers=n_layers, n_units=n_units, name=model_name + '_ptr_net_2')

        self.transformer1.build_model()
        self.enc_out1 = self.transformer1.enc_out

        self.transformer2.build_model()
        self.enc_out2 = self.transformer2.enc_out

        self.ptr_net1.decoder(memory=self.enc_out1, num_layers=2, memory_mask=memory_mask_1)
        self.ptr_net2.decoder(memory=self.enc_out2, num_layers=2, memory_mask=memory_mask_2)

        with tf.variable_scope(model_name):
            # self.thought_vector = tf.concat([self.enc_out1, self.enc_out2], axis=1)
            with tf.variable_scope('hidden_state'):
                # self.thought_vector = self.enc_out1
                self.thought_vector = tc.layers.layer_norm(self.enc_out1, center=True, scale=True)
                # self.cell = cell(num_units=128)
                # outputs, state = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.thought_vector, dtype=tf.float32)
                # hid_state = tf.concat(tf.unstack(outputs, axis=1), axis=-1)    争取搞清楚为什么此处用RNN会出错

                # 从三维张量转换为二维张量，为了方便后续输出头直接用全连接层
                hid_state = tf.concat(tf.unstack(self.thought_vector, axis=1), axis=-1)
                hid_state = tf.layers.dense(inputs=hid_state, units=512, activation=None, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))

            # 二维的，决定是飞还是打；第一维飞，第二维打
            # 决定每一个 batch 中 out_head1_action 所指定的两个动作是否都可用，至少飞是肯定可以选的
            with tf.variable_scope('head1_out'):
                out_head1_logits = tf.layers.dense(inputs=hid_state, units=2, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                self.out_head1_logits = out_head1_logits + tf.cast(action_mask_1, tf.float32)
                out_head1_dist = Categorical(logits=self.out_head1_logits)
                self.out_head1_action = out_head1_dist.sample()

            # 如果是飞(out_head1_action == 0)，选取航向角；如果是打(out_head1_action == 1)，选取敌方单位。
            with tf.variable_scope('head2_out'):
                out_head2_logits = tf.layers.dense(inputs=hid_state, units=10, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                out_head2_prob = tf.nn.softmax(out_head2_logits, axis=1)
                out_head2_dist = Categorical(logits=out_head2_logits)
                out_head2_ = out_head2_dist.sample()
                
                # 第三个输出头
                self.out_head2_action = out_head2_
                self.out_head2_prob = out_head2_prob

                self.out_head2_action = tf.where(
                    tf.cast(self.out_head1_action, tf.bool),
                    tf.squeeze(self.ptr_net2.pointers, axis=0),
                    out_head2_)

                gather_index = tf.concat([
                    tf.expand_dims(tf.range(out_head2_prob.shape.as_list()[0]), axis=1), 
                    tf.expand_dims(out_head2_, axis=1)], axis=1)

                self.out_head2_prob = tf.where(
                    tf.cast(self.out_head1_action, tf.bool),
                    tf.reduce_max(tf.squeeze(self.ptr_net2.pointer_prob, axis=0), axis=-1),
                    tf.gather_nd(out_head2_prob, gather_index))

            with tf.variable_scope('value'):
                value = tf.layers.dense(inputs=hid_state, units=32, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0))
                # value = tc.layers.layer_norm(value, center=True, scale=True)
                self.value_fore = value
                self.value = tf.squeeze(tf.layers.dense(inputs=value, units=1, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0)), axis=-1)
                

    def output(self):
        self.batch_first_pointers = tf.transpose(self.ptr_net1.pointers, [1, 0])
        self.batch_first_pointer_prob = tf.transpose(self.ptr_net1.pointer_prob, [1, 0, 2])
        return self.batch_first_pointers, self.batch_first_pointer_prob, self.out_head1_action, self.out_head1_logits, self.out_head2_action, self.out_head2_prob, self.value, self.thought_vector, self.value_fore
                

if __name__ == '__main__':
    _raw_input1 = tf.placeholder(dtype=tf.float32, shape=[10, 8, 4], name='input1')
    _raw_input2 = tf.placeholder(dtype=tf.float32, shape=[10, 8, 4], name='input2')
    net = Network(_raw_input1=_raw_input1, _raw_input2=_raw_input2)

    sess = tf.Session()
    # input_data = np.random.randint(low=1, high=16, size=[2, 3, 4])
    input_data = np.random.normal(size=[10, 8, 4])
    memory_mask = np.ones(shape=[9, 8])
    memory_mask_part2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
    memory_mask = np.concatenate([memory_mask, memory_mask_part2], axis=0)

    init = tf.global_variables_initializer()
    sess.run(init)
    res1, res2, res3 = sess.run([net.ptr_net2.decoder_outputs, net.ptr_net2.pointer_prob, net.ptr_net2.final_sample_id_ta], 
    feed_dict={
        _raw_input1: input_data, 
        _raw_input2: input_data, 
        net.ptr_net1.memory_mask: memory_mask,
        net.ptr_net2.memory_mask: memory_mask,
    })

    print('res1 decoder_outputs:', res1)
    print('res2 pointer prob:', res2)

    # print(np.isnan(res2[3, 9, 0]))
    # print('res3:', res3)
    # print('succeed !')
