import numpy as np 
import tensorflow as tf  
import gym  
from tensorflow.distributions import Categorical, Normal
from policy import PPO

gamma = 0.99 

class Train(object):
    def __init__(self):
        self.policy = PPO()
        self.sess = tf.Session()

        self.build_summary()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=3)

    
    def build_summary(self):
        # 此处添加 tensorboard 所需要记录的标量
        tf.summary.scalar('loss', self.loss)

        self.merged = tf.summary.merge_all()
        # 指定一个文件来保存图，第一个参数给定的是地址，通过调用 tensorboard --logdir=logs 查看 tensorboard
        self.train_writer = tf.summary.FileWriter('train_log', self.sess.graph) 

    def output_summary(self):
        rew_summ = self.sess.run([self.merged], feed_dict=self.feed_dict)
        return rew_summ 

    def train(self):
        _, summary, loss = self.sess.run([self.train_op, self.merged, self.loss], feed_dict=self.feed_dict)



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(1)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    ppo = PPO
    
    MAX_EPISODE = 4000 

    for episode in range(MAX_EPISODE):
        obs = env.reset()
        rewards = []
        t = 0 
        value_buffer, reward_buffer, act_buffer, obs_buffer, next_obs_buffer = [], [], [], [], [] 
        while True:
            t = t + 1
            obs = obs[None, :]
            act, value = ppo.select_action(obs)
            obs = obs.squeeze(axis=0)
            obs_next, reward, done, info = env.step(act)

            if done:
                value_buffer.append(value)
                reward_buffer.append(reward)
                obs_buffer.append(obs)
                act_buffer.append(act)
                rewards.append(reward)
                next_obs_buffer.append(obs_next)
                break 

            value_buffer.append(value)
            reward_buffer.append(reward)
            obs_buffer.append(obs)
            act_buffer.append(act)
            rewards.append(reward)
            next_obs_buffer.append(obs_next)

            obs = obs_next 

        value_buffer.append(0)
        rewards_to_go = discounted_cumulative_sum(reward_buffer, gamma)
        delta = np.array(reward_buffer) + bp.array(value_buffer[1:]) * gamma - np.array(value_buffer[:-1])

        adv_buffer = np.array(discounted_cumulative_sum(delta, gamma * 0.95, 0))
        adv_buffer = (adv_buffer - np.mean(adv_buffer))/ (np.std(adv_buffer) + 1e-8)

        obs_buffer = np.array(obs_buffer)
        act_buffer = np.array(act_buffer)
        ret_buffer = np.array(rewards_to_go)

        ppo.data_feed(obs_buffer, act_buffer, adv_buffer, ret_buffer, sum(reward_buffer))
        loss = ppo.train()

        rew_summ = ppo.summary()
        ppo.train_writer.add_summary(rew_summ, episode)

        print("episode: {:5}, reward: {:5}, total loss: {:8.2f}".format(episode, episode, episode))

