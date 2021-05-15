import tensorflow as tf
import numpy as np 

class QMIXNet(object):
    def __init__(self, act_dim, agent_num, num_units, name):
        self.act_dim = act_dim  # 每个智能体的动作空间的维度
        self.agent_num = agent_num  # 共有多少个智能体
        self.num_units = num_units
        self.name = name 
        
    def build(self, feature_embedding):
        # agent_state 的输入维度为[num_agents, num_units, feature_embedding]，之所以有第一个维度是因为目前所有单位共用网络，因此可以放在一起处理
        with tf.variable_scope(self.name):
            self.agent_state = tf.placeholder(
                dtype=tf.float32, shape=[None, self.agent_num, self.num_units, feature_embedding], name='agent_state')
            # self.global_state = tf.placeholder(dtype=tf.float32, shape=[None, num_units, feature_embedding], name='global_state')
            self.global_state = tf.placeholder(dtype=tf.float32, shape=[None, self.num_units, feature_embedding], name='global_state')
            self.act_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.agent_num], name='action_each_agent')
            self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')

        self.build_agent_net()
        self.build_mix_net()
        self.build_total_q_value()

    def build_agent_net(self):
        with tf.variable_scope(self.name):
            with tf.variable_scope('agent_net'):
                out = tf.layers.dense(
                    inputs=self.agent_state, 
                    units=32, 
                    activation=tf.nn.sigmoid, 
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.1))
                self.q_value = tf.reduce_mean(tf.layers.dense(
                    inputs=out, 
                    units=self.act_dim, 
                    activation=tf.nn.relu, 
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.1)), axis=2)
                self.select_action = tf.argmax(self.q_value, axis=2)

    def build_mix_net(self):
        with tf.variable_scope(self.name):
            with tf.variable_scope('hyper_net'):
                self.mix_param_1 = tf.layers.dense(
                    inputs=self.global_state, 
                    units=self.agent_num, 
                    activation=tf.nn.relu6,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.1))
                self.mix_param_2 = tf.layers.dense(
                    inputs=self.global_state, 
                    units=1, 
                    activation=tf.nn.relu6,
                    kernel_initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.1))

    def build_total_q_value(self):
        with tf.variable_scope(self.name):
            with tf.variable_scope('total_q_value'):
                q_value = tf.reduce_max(self.q_value, axis=2, keep_dims=True)
                out = tf.matmul(self.mix_param_1, q_value)
                out = tf.transpose(out, [0, 2, 1])
                self.total_q_value = tf.matmul(out, self.mix_param_2)
                self.total_q_value = tf.squeeze(tf.squeeze(self.total_q_value, axis=-1), axis=-1)


class RLFighter(object):
    def __init__(
        self, 
        num_act,
        agent_num,
        num_units,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=25,
        memory_size=500,
        batch_size=32,
        e_greedy_increment=None):

        self.num_act = num_act
        self.agent_num = agent_num
        self.num_units = num_units
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size 
        self.batch_size = batch_size 
        self.e_greedy_increment = e_greedy_increment

        self.s_memory = []
        self.global_s_memory = []
        self.a_memory = []
        self.done_memory = []
        self.r_memory = []
        self.next_s_memory = []
        self.global_next_s_memory = []
        self.update_counter = 0

        # self.behavior_net = QMIXNet(act_dim=self.num_act, agent_num=self.agent_num, num_units=self.num_units, name='behavior_net')
        # self.target_net = QMIXNet(act_dim=self.num_act, agent_num=self.agent_num, num_units=self.num_units, name='target_net')
        self.sess = tf.Session()
        
    def init(self, feature_embedding):
        self.behavior_net.build(feature_embedding)
        self.target_net.build(feature_embedding)
        self.build_loss()

        self.sess.run(tf.global_variables_initializer())

        self.behavior_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='behavior_net')
        self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

    def choose_action(self, agent_obs, global_obs):
        value, action = self.sess.run([self.behavior_net.q_value, self.behavior_net.select_action], feed_dict={
            self.behavior_net.agent_state: agent_obs, 
            self.behavior_net.global_state: global_obs})
        return action, value

    def store_transition(self, s, global_s, a, r, s_, global_next_s):
        self.s_memory.append(s)
        self.global_s_memory.append(global_s)
        self.a_memory.append(a)
        self.r_memory.append(r)
        self.next_s_memory.append(s_)
        self.global_next_s_memory.append(global_next_s)
        # self.memory_counter += 1

    def build_loss(self):
        target_value = self.target_net.total_q_value
        self.target = self.behavior_net.reward_ph + self.reward_decay * target_value

        curr_q_value = self.behavior_net.total_q_value
        self.loss = tf.reduce_sum(tf.square(curr_q_value - tf.stop_gradient(self.target)))
        self.train_op = tf.train.AdamOptimizer(0.005).minimize(self.loss)
    
    def update_param(self):
        assign_op = [tf.assign(t, b) for t, b in zip(self.target_params, self.behavior_params)]
        self.sess.run(assign_op)

    def learn(self):
        self.s_memory = np.array(self.s_memory)
        self.global_s_memory = np.array(self.global_s_memory)
        self.a_memory = np.array(self.a_memory)
        self.r_memory = np.array(self.r_memory)
        self.next_s_memory = np.array(self.next_s_memory)
        self.global_next_s_memory = np.array(self.global_next_s_memory)

        loss, _, mix_param_1, mix_param_2, value1, value2 = self.sess.run(
            [self.loss, self.train_op, self.behavior_net.mix_param_1, self.behavior_net.mix_param_2, self.target, self.behavior_net.total_q_value],
            feed_dict={
                self.behavior_net.agent_state: self.s_memory,
                self.behavior_net.global_state: self.global_s_memory,
                self.behavior_net.reward_ph: self.r_memory,
                self.target_net.agent_state: self.next_s_memory,
                self.target_net.global_state: self.global_next_s_memory,
            })

        # print('action', self.a_memory)
        # print('loss:', loss)
        print('value1', value1)
        # print('value2:', value2)
        # print('mex_param_1', mix_param_1)
        # print('state:', self.s_memory[0][0])
        # print('next state', self.next_s_memory[0][0])
        # print('reward memory: ', self.r_memory)
        # print('next_s_memory', self.next_s_memory)

        self.update_counter += 1
        if self.update_counter % self.replace_target_iter == 0:
            self.update_param()
        
        self._clear_memory()
    
    def _clear_memory(self):
        self.s_memory = []
        self.global_s_memory = []
        self.a_memory = []
        self.done_memory = []
        self.r_memory = []
        self.next_s_memory = []
        self.global_next_s_memory = []


