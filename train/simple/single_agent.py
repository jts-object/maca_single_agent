from net_module.network import Network
import tensorflow as tf
from tensorflow.distributions import Categorical
import numpy as np 
import os

class RLFighter(object):
    def __init__(
        self, 
        learning_rate=1e-5,
        reward_decay=0.9,
        batch_size=32,
        n_pointers=3,
        clip_param=0.2):

        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.batch_size=batch_size
        self.n_pointers = n_pointers
        self.clip_param = clip_param
        self.vf_loss_coeff = 0.2
        
        self.s_memory = []
        self.a_memory = []
        self.done_memory = []
        self.r_memory = []
        self.next_s_memory = []
        self.update_counter = 0

        self.init_ph()
        
        self.infer_net = Network( 
            _raw_input1=self.infer_obs_ph, 
            _raw_input2=self.infer_enobs_ph,
            memory_mask_1=self.infer_mem_mask1, 
            memory_mask_2=self.infer_mem_mask2, 
            action_mask_1=self.infer_action_mask,
            n_pointers=3,
            batch_size=1,
            seq_max_len=10, # 因为有 10 个单位
            model_name='infer_model')
        self.train_net = Network(
            _raw_input1=self.train_obs_ph, 
            _raw_input2=self.train_enobs_ph,
            memory_mask_1=self.train_mem_mask1, 
            memory_mask_2=self.train_mem_mask2, 
            action_mask_1=self.train_action_mask,
            n_pointers=3,
            batch_size=self.batch_size,
            seq_max_len=10, # 因为有 10 个单位
            model_name='train_model')
        self.get_model_output()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.init()
        
    def init(self):
        self.build_loss()
        self.build_summary()
        self.sess.run(tf.global_variables_initializer())

        # self.behavior_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='behavior_net')
        # self.target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

    def init_ph(self):
        """ 推断用的占位符 """
        self.infer_obs_ph = tf.placeholder(dtype=tf.float32, shape=[1, 10, 7], name='infer_obs')
        self.infer_enobs_ph = tf.placeholder(dtype=tf.float32, shape=[1, 10, 7], name='infer_enobs')
        self.infer_mem_mask1 = tf.placeholder(dtype=tf.float32, shape=[10], name='infer_mem_mask1')
        self.infer_mem_mask2 = tf.placeholder(dtype=tf.float32, shape=[10], name='infer_mem_mask2')
        self.infer_action_mask = tf.placeholder(dtype=tf.float32, shape=[2], name='infer_act_mask')
        
        """ 训练用的占位符 """
        self.train_obs_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 10, 7], name='train_obs')
        self.train_enobs_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 10, 7], name='train_enobs')
        self.train_mem_mask1 = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 10], name='train_mem_mask1')
        self.train_mem_mask2 = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 10], name='train_mem_mask2')
        self.train_action_mask = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 2], name='train_act_mask')
        
        self.action_head0_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.n_pointers], name='action_head0')
        self.action_head1_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, ], name='action_head1')
        self.action_head2_ph = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, ], name='action_head2')
        self.act_prob_head0_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 3, 10], name='act_prob_head0')
        self.act_logits_head1_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 2], name='act_logits_head1')
        self.act_prob_head2_ph = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,], name='act_prob_head2')

        self.advantage_ph = tf.placeholder(dtype=tf.float32, shape=[None, ], name='advantage')
        self.behavior_value_ph = tf.placeholder(dtype=tf.float32, shape=[None, ], name='behavior_value')
        self.target_value_ph = tf.placeholder(dtype=tf.float32, shape=[None, ], name='target_value')
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None, ], name='reward')
        
    def get_model_output(self):
        self.infer_head0_pointers, self.infer_head0_pointer_prob, self.infer_head1_action, self.infer_head1_logits, self.infer_head2_action, self.infer_head2_prob, self.infer_value, _ = self.infer_net.output()
        self.head0_pointers, self.head0_pointer_prob, self.head1_action, self.head1_logits, self.head2_action, self.head2_prob, self.value, self.vector = self.train_net.output()
    
    def choose_action(self, obs, en_obs, mem_mask1, mem_mask2, action_mask):
        head0_pointers, head0_pointer_prob, head1_act, head1_logits, head2_act, head2_prob, value = self.sess.run([
            self.infer_head0_pointers, 
            self.infer_head0_pointer_prob, 
            self.infer_head1_action,
            self.infer_head1_logits,
            self.infer_head2_action, 
            self.infer_head2_prob, 
            self.infer_value], feed_dict={
                self.infer_obs_ph: obs,
                self.infer_enobs_ph: en_obs,
                self.infer_mem_mask1: mem_mask1,
                self.infer_mem_mask2: mem_mask2,
                self.infer_action_mask: action_mask,
        })

        return head0_pointers, head0_pointer_prob, head1_act, head1_logits, head2_act, head2_prob, value
    
    def build_summary(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('reward', self.reward_ph)
        self.merged = tf.summary.merge_all()

        # 指定一个文件来保存图，第一个参数给定的是地址，通过调用 tensorboard --logdir=$(pwd) 查看 tensorboard
        self.train_writer = tf.summary.FileWriter('train_log', self.sess.graph)

    def output_summ(self):
        summ = self.sess.run(self.merged, feed_dict=self.feed_dict)
        return summ

    def build_loss(self):
        # 当前 behavior_action_head0_prob 的维度为 [batch, n_pointers, num_units]，第一步需要得到在新策略之下选出 n_pointers 个单位的概率
        
        range_ = tf.range(self.head0_pointer_prob.shape.as_list()[1])
        range_ = tf.tile(tf.expand_dims(range_, axis=0), [self.head0_pointer_prob.shape.as_list()[0], 1])
        index = tf.stack([range_, self.head0_pointers], axis=0)

        range_ = tf.range(self.head0_pointer_prob.shape.as_list()[0])
        range_ = tf.tile(tf.expand_dims(range_, axis=1), [1, self.head0_pointer_prob.shape.as_list()[1]])
        res = tf.concat([tf.expand_dims(range_, axis=0), index], axis=0)
        final_index = tf.transpose(res, [1, 2, 0])
        
        head0_prob = tf.gather_nd(self.head0_pointer_prob, final_index)
        neglogp_head0 = tf.clip_by_value(-tf.log(tf.reduce_prod(head0_prob, axis=1)), -1.e6, 1.e6)     # 最后维度: [batch]
        
        self.new_neglogp_head0 = neglogp_head0  # debug
        # 得到当前动作在旧策略下的概率
        range_ = tf.range(self.head0_pointer_prob.shape.as_list()[1])
        range_ = tf.tile(tf.expand_dims(range_, axis=0), [self.head0_pointer_prob.shape.as_list()[0], 1])
        index = tf.stack([range_, self.head0_pointers], axis=0)

        range_ = tf.range(self.head0_pointer_prob.shape.as_list()[0])
        range_ = tf.tile(tf.expand_dims(range_, axis=1), [1, self.head0_pointer_prob.shape.as_list()[1]])
        res = tf.concat([tf.expand_dims(range_, axis=0), index], axis=0)
        final_index = tf.transpose(res, [1, 2, 0])
        
        head0_prob = tf.gather_nd(self.act_prob_head0_ph, final_index)     # self.act_prob_head0 存储的是旧策略对应的概率
        old_neglogp_head0 = tf.clip_by_value(-tf.log(tf.reduce_prod(head0_prob, axis=1)), -1.e6, 1.e6)     # 维度: [batch]

        self.old_neglogp_head0 = old_neglogp_head0  # debug

        
        # 第二步得到 head1 的输出概率
        # # 得到在当前网络输出之下的动作概率
        # behavior_dist = Categorical(logits=self.head1_logits)
        # neglogprob_head1 = tf.clip_by_value(-tf.log(behavior_dist.prob(self.head1_action)), -1.e5, 1.e5)
        neglogprob_head1 = -tf.reduce_max(self.head1_logits, axis=-1)
        
        # old_dist = Categorical(logits=self.act_logits_head1_ph)
        # old_neglogprob_head1 = tf.clip_by_value(-tf.log(old_dist.prob(self.action_head1_ph)), -1e5, 1.e5)
        old_neglogprob_head1 = -tf.reduce_max(self.act_logits_head1_ph, axis=-1)

        self.new_neglogp_head1 = neglogprob_head1    # debug
        self.old_neglogp_head1 = old_neglogprob_head1    # debug

        self.new_logits_head1 = self.head1_logits   # debug
        self.old_logits_head1 = self.act_logits_head1_ph    # debug

        self.old_neglogp = tf.stop_gradient(old_neglogp_head0 + old_neglogprob_head1)
        self.neglogp = neglogp_head0 + neglogprob_head1

        
        ratio = tf.clip_by_value(tf.exp(self.old_neglogp - self.neglogp), -1e4, 1e4)
        self.ratio = ratio  # debug

        advantage = self.target_value_ph - self.value
        mean, var = tf.nn.moments(advantage, axes=[0])
        advantage = (advantage - mean) / (tf.sqrt(var) + 1e-8)
        advantage = tf.stop_gradient(advantage) # 为何 advantage 不需要导致梯度的反向传播。应该是为了防止重复计算吧

        self.advantage = advantage  # debug
        # ppo loss
        surr1 = ratio * advantage
        surr2 = tf.clip_by_value(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
        # surr = tf.minimum(surr1, surr2) 
        surr = surr2    # debug 

        self.surr1 = surr1  # debug
        self.surr2 = surr2  # debug
        self.surr = surr    # debug

        clip_prob = tf.reduce_mean(tf.where(surr1 > surr2, tf.ones_like(surr1), tf.zeros_like(surr1)))
        ratio_diff = tf.reduce_mean(tf.abs(ratio - 1.))

        self.policy_loss = - tf.reduce_mean(surr)
        # value_loss_1 = - tf.reduce_mean(value * advantage)

        # vpredclipped 是为了保证与 behavior_value 差距不是太远，增加了训练的稳定性
        vpredclipped = self.behavior_value_ph + tf.clip_by_value(self.value - self.behavior_value_ph,
                                                        -self.clip_param,
                                                        self.clip_param)
        vf_losses1 = tf.square(self.value - self.target_value_ph)
        vf_losses2 = tf.square(vpredclipped - self.target_value_ph)
        # self.value_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        self.value_loss = vf_losses1

        # entropy
        # entropy = distribution_head0.entropy(behavior_action_head0)
        # mean_entropy = tf.reduce_mean(entropy)

        # loss = policy_loss + value_loss * self.vf_loss_coeff - mean_entropy * self.entropy_coeff
        self.loss = self.policy_loss + self.value_loss * self.vf_loss_coeff

        # 测试不同优化器的影响
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.2)
        # optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)
        self.grads = [grad for grad, _ in gvs if grad is not None]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) # debug，所有可训练参数
        for grad, var in gvs:
            if grad is not None:
                print('grad:', grad)
                print('var:', var)
        

        cliped_gvs = [(tf.clip_by_norm(grad, 2.), var) for grad, var in gvs if grad is not None]

        self.cliped_gvs = cliped_gvs
        self.train_op = optimizer.apply_gradients(cliped_gvs)

    # def update_param(self):
    #     assign_op = [tf.assign(t, b) for t, b in zip(self.target_params, self.behavior_params)]
    #     self.sess.run(assign_op)

    def learn(
        self, obs_list, en_obs_list, next_obs_list, next_en_obs_list, mem_mask1_list, mem_mask2_list, act_mask_list,
        reward_list, head0_act_list, head1_act_list, head2_act_list, head0_prob_list, head1_logits_list, head2_prob_list, behavior_value_list):

        obs_array = np.array(obs_list)
        en_obs_array = np.array(en_obs_list)
        next_obs_array = np.array(next_obs_list)
        next_en_obs_array = np.array(next_en_obs_list)
        mem_mask1_array = np.array(mem_mask1_list)
        mem_mask2_array = np.array(mem_mask2_list)
        act_mask_array = np.array(act_mask_list)
        reward_array = np.array(reward_list)
        head0_act_array = np.squeeze(np.array(head0_act_list), axis=1)
        head1_act_array = np.squeeze(np.array(head1_act_list), axis=-1)
        head2_act_array = np.squeeze(np.array(head2_act_list), axis=-1)
        head0_prob_array = np.squeeze(np.array(head0_prob_list), axis=1)
        head1_logits_array = np.squeeze(np.array(head1_logits_list), axis=1)
        head2_prob_array = np.squeeze(np.array(head2_prob_list), axis=-1)
        behavior_value_array = np.squeeze(np.array(behavior_value_list), axis=-1)

        self.feed_dict = {
                self.train_obs_ph: obs_array,
                self.train_enobs_ph: en_obs_array,
                self.train_mem_mask1: mem_mask1_array,
                self.train_mem_mask2: mem_mask2_array,
                self.train_action_mask: act_mask_array,
                self.action_head0_ph: head0_act_array,
                self.action_head1_ph: head1_act_array,
                self.action_head2_ph: head2_act_array,
                self.act_prob_head0_ph: head0_prob_array,
                self.act_logits_head1_ph: head1_logits_array,
                self.act_prob_head2_ph: head2_prob_array,
                self.behavior_value_ph: behavior_value_array,
                self.target_value_ph: behavior_value_array,
                self.reward_ph: reward_array,
            }

        _, new_head0_prob, new_logp_head0, old_logp_head0, new_logp_head1, old_logp_head1, ratio, adv, surr1, surr2, surr, new_logit, old_logit, p_loss, loss, v_loss, value, target_v, vector, grads, varss = self.sess.run(
            [
                self.train_op, self.head0_pointer_prob, self.new_neglogp_head0, self.old_neglogp_head0, self.new_neglogp_head1, self.old_neglogp_head1, 
                self.ratio, self.advantage, self.surr1, self.surr2, self.surr, self.new_logits_head1, self.old_logits_head1, self.policy_loss, self.loss, 
                self.value_loss, self.value, self.target_value_ph, self.vector, self.grads, self.var_list], feed_dict=self.feed_dict)
        
        # print('vector: ', vector[0])
        # print('new_head0_prob: ', new_head0_prob)
        # print('new_logp_head0: ', new_logp_head0)
        # print('old_logp_head0: ', old_logp_head0)
        print('new logit :', new_logit)
        print('old_logit : ', old_logit)
        # print('new_logp_head1: ', new_logp_head1)
        # print('old_logp_head1: ', old_logp_head1)
        print('ratio: ', ratio)
        # print('adv: ', adv)
        # print('surr1: ', surr1)
        # print('surr2: ', surr2)
        # print('surr: ', surr)
        # print('value loss: ', v_loss)
        # print('policy loss: ', p_loss)
        print('loss: ', loss)
        print('value:', value)
        print('target_value:', target_v)
        
        from tensorflow.python.util import nest
        
        for item1, item2 in zip(nest.flatten(varss), nest.flatten(grads)):
            # print('item1:', item1)
            print('item2:', item2)
        
       
    def save_model(self, model_path, model_name, iterations):
        self.saver.save(self.sess, os.path.join(model_path, model_name + iterations))



