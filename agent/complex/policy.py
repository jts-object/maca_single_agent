import tensorflow as tf 
from agent.complex.network import Network 


class PPO:
    def __init__(self, act_head_list, obs):
        """
        act_head_list: 每个动作输出头的维度大小
        
        """
        self.old_model = Network(obs, act_head_list, name='old_policy', trainable=False)
        self.new_model = Network(obs, act_head_list, name='new_policy', trainable=True)
        self.dist_new_policy_list, self.new_value, self.new_params = self.new_model.output
        self.dist_old_policy_list, self.old_value, self.old_params  = self.old_model.output
        self.num_heads = len(act_head_list)
        self.action_list = [tf.placeholder(dtype=tf.int32, shape=[
            None, 1], name='action_head_{}'.format(i)) for i in range(self.num_heads)]
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None,], name='advantage')
        self.dis_rew = tf.placeholder(dtype=tf.float32, shape=[None, ], name='discounted_reward')

        # self.init = tf.global_variables_initializer()
        self.update_op = [old_p.assign(p) for p, old_p in zip(self.old_params, self.new_params)]
        
    def sample_action(self, ):
        # 对每个动作头输出进行采样
        with tf.variable_scope('sample_action'):
            act_list = []
            for dist in self.dist_new_policy_list:
                act_list.append(dist.sample())
        return act_list


    def build_loss(self):
        # 对所有动作输出头计算损失
        with tf.variable_scope('policy_loss'):
            new_log_prob = 0.
            old_log_prob = 0.
            for i, act in enumerate(self.action_list):
                new_log_prob += self.dist_new_policy_list[i].log_prob(act)
                old_log_prob += self.dist_old_policy_list[i].log_prob(act)

            ratio = tf.exp(new_log_prob - old_log_prob)
            surr = ratio * self.advantage
            self.policy_loss = -tf.reduce_mean(tf.minimum(
                surr, 
                tf.clip_by_value(ratio, 1. - 0.2, 1. + 0.2) * self.advantage))
        
        with tf.variable_scope('value_loss'):
            self.value_loss = tf.reduce_mean(tf.square(self.new_value - self.dis_rew))
        
        with tf.variable_scope('entropy_loss'):
            self.entropy_loss = 0. 

        with tf.variable_scope('total_loss'):
            self.loss = tf.add(self.value_loss, self.policy_loss)

        with tf.variable_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    
    def learn(self, sess, feed_dict):
        self.build_loss()

        total_loss, _, value = sess.run([self.loss, self.train_op, self.new_value], feed_dict=feed_dict)
        sess.run(self.update_op)

