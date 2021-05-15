import tensorflow as tf
import numpy as np 
# from agent.complex.transformer import Transformer 
from tensorflow.distributions import Categorical


class Network(object):
    def __init__(self, obs, act_head_list, name, trainable=True, hidden_sizes=(256, 256, 1), activation=tf.nn.relu, output_activation=None):

        """Initialize the Network.
        Args:
            obs: tf placeholer, the observation got from environment.
            hidden_sizes: tuple, the dimensions of the hidden layers.
            name: name of network.
            activation: tf activation function before the output layer.
            output_activation: tf activation function of the output layer.
        """

        self.act_head_list = act_head_list      # 一个列表，每个元素代表网络的每个动作输出头的维度
        # self.trans_net = Transformer(_raw_input=self.raw_input, d_model=32)
        self.action_head = len(self.act_head_list)
        self.name = name
        self.build(obs, hidden_sizes, activation, output_activation, trainable)
        
    def build(self, obs, hidden_sizes, activation, output_activation, trainable):
        # self.trans_net.build_model()
        # self.encoder_out = self.trans_net.enc_out
        with tf.variable_scope(self.name):
            middle_out = self.mlp(obs, hidden_sizes=hidden_sizes, activation=activation, output_activation=output_activation, trainable=trainable)
        
            self.final_out = {}
            for i, act_dim in enumerate(self.act_head_list):
                with tf.variable_scope('policy_net_head_{}'.format(i)):
                    tmp_logits = tf.squeeze(middle_out, axis=-1)
                    tmp_logits = tf.layers.dense(inputs=tmp_logits, units=act_dim, activation=None, trainable=trainable)
                self.final_out['policy_net_head_{}'.format(i)] = tmp_logits
        
            with tf.variable_scope('value_net'):
                self.value = tf.squeeze(self.mlp(middle_out, list(hidden_sizes), activation, output_activation, trainable=trainable), axis=-1)
                self.value = tf.reduce_mean(self.value, axis=-1)

    def mlp(self, x, hidden_sizes, activation, output_activation, trainable):
        with tf.variable_scope('mlp'):
            for h in hidden_sizes[:-1]:
                x = tf.layers.dense(inputs=x, units=h, activation=activation, trainable=trainable)
            return tf.layers.dense(inputs=x, units=hidden_sizes[-1], activation=output_activation, trainable=trainable)

    def output(self):
        head_dist = {'policy_net_head_{}'.format(i): Categorical(logits=logits) for i, logits in enumerate(self.final_out.values())}
        action_sample = []      # 需要在此对动作进行采样吗？
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        return head_dist, self.value, params


if __name__ == '__main__':
    input_obs = tf.placeholder(dtype=tf.float32, shape=[10, 5, 4], name='input')
    net = Network(obs=input_obs, act_head_list=[3, 4], name='network')

    head_dist, val, param = net.output()
    print(head_dist)
    print(val.shape)
    print(param)
