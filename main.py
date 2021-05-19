#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: main.py
@time: 2018/7/25 0025 10:01
@desc: 
"""

import os
import copy
import numpy as np
import sys 
# sys.path.append('../../')
from agent.fix_rule_no_att.agent import Agent
from interface import Environment
from train.simple import dqn
from train.simple import qmix
from train.simple import single_agent
import time
MAP_PATH = 'maps/1000_1000_fighter10v10.map'

RENDER = False
MAX_EPOCH = 1000
BATCH_SIZE = 50
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 10         # 不同角度的个数
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1 # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM      # 总的动作数
LEARN_INTERVAL = 100
SAVE_MODEL_INTERVAL = 100 # 保存模型的间隔


def replace_rowvec(matrix, ind):
    "替换输入matrix中的第ind行并返回"
    tmp_mat = copy.deepcopy(matrix)
    curr_row = np.zeros_like(tmp_mat[ind, :])
    curr_row[2], curr_row[3] = tmp_mat[ind, 2], tmp_mat[ind, 3]     # x, y 坐标的位置
    
    # 将全局观测的前一半，也就是我方态势的坐标减去相应坐标得到相对坐标
    tmp_num = int(tmp_mat.shape[0]/2)
    tmp_mat = np.concatenate([tmp_mat[0: tmp_num, :] - curr_row, tmp_mat[tmp_num:, :]], axis=0)
    num_column = tmp_mat.shape[1]
    tmp_mat[ind, :] = np.full([num_column], -1.)
    
    return tmp_mat

if __name__ == "__main__":
    blue_agent = Agent()
    # blue_agent = FixRuleNoAttAgent()
    # get agent obs type
    red_agent_obs_ind = 'simple'
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    red_detector_action = []
    fighter_model = single_agent.RLFighter(batch_size=BATCH_SIZE)
    
    # 模型保存相关参数
    train_epoch = 0
    model_path = 'model/'
    model_name = 'model.ckpt'
    time_to_save = False

    # execution
    for x in range(MAX_EPOCH):
        step_cnt = 0
        num_agents = 10
        num_units = 20
        env.reset()

        obs_list, en_obs_list = [], []
        behavior_value_list = []
        action_head0_list, action_head1_list, action_head2_list = [], [], []
        probs_head0_list, logits_head1_list, probs_head2_list = [], [], []
        memory_mask1_list, memory_mask2_list, action_mask_list = [], [], []
        next_obs_list, next_en_obs_list = [], []
        reward_list = []

        while True:
            red_detector_action, red_fighter_action = [], []
            red_obs_dict, blue_obs_dict = env.get_obs()

            fighter_tmp_obs = np.asarray(red_obs_dict['fighter'], dtype=np.float32)   # 在同质化的设定中，只有 fighter
            enemy_tmp_obs = np.asarray(red_obs_dict['enemy'], dtype=np.float32)
            fighter_visible_enemys_dict = red_obs_dict['fighter_visible_enemys_dict']
            fighter_data_obs_list = red_obs_dict['fighter_raw']
            # get obs
            if step_cnt == 0:
                red_obs_dict, blue_obs_dict = env.get_obs()     # 此处得到 obs
            # get red and blue action
            blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step_cnt)
            obs_got_ind = [False] * red_fighter_num

            red_obs, red_en_obs = np.asarray(red_obs_dict['fighter']), np.asarray(red_obs_dict['enemy'])
            agent_obs = np.zeros([num_agents] + list(red_obs.shape))

            for y in range(red_fighter_num):
                if red_obs_dict['alive'][y]:
                    obs_got_ind[y] = True
            
            def generate_mask(obs_got_ind, enemy_tmp_obs):
                mask1 = np.asarray([np.int(item) for item in obs_got_ind])    # 我方存活单位的掩码
                mask2 = np.where(enemy_tmp_obs[:, 0], np.ones_like(enemy_tmp_obs[:, 0], dtype=np.int32), np.zeros_like(enemy_tmp_obs[:, 0], dtype=np.int32))    # 敌方存活单位的掩码
                if np.sum(mask2) > 0.:        # 如果敌方有发现单位，表明可以采取打这个动作
                    action_mask = np.array([0., 0.])
                else:
                    action_mask = np.array([0., -1.e6])
            
                return mask1, mask2, action_mask

            mem_mask1, mem_mask2, act_mask = generate_mask(obs_got_ind, red_en_obs)
            red_obs, red_en_obs = np.expand_dims(red_obs, axis=0), np.expand_dims(red_en_obs, axis=0)
        
            head0_pointers, head0_pointer_prob, head1_act, head1_logits, head2_act, head2_prob, behavior_value = fighter_model.choose_action(red_obs, red_en_obs, mem_mask1, mem_mask2, act_mask)
            head0_pointers, head0_pointer_prob, head1_act = head0_pointers.tolist(), head0_pointer_prob.tolist(), head1_act.tolist()
            head1_logits, head2_act, head2_prob, behavior_value = head1_logits.tolist(), head2_act.tolist(), head2_prob.tolist(), behavior_value.tolist()
        
            obs_list.append(red_obs_dict['fighter'].tolist())
            en_obs_list.append(red_obs_dict['enemy'].tolist())
            
            behavior_value_list.append(behavior_value)
            action_head0_list.append(head0_pointers)
            action_head1_list.append(head1_act)
            action_head2_list.append(head2_act)

            probs_head0_list.append(head0_pointer_prob)
            logits_head1_list.append(head1_logits)
            probs_head2_list.append(head2_prob)

            memory_mask1_list.append(mem_mask1)
            memory_mask2_list.append(mem_mask2)
            action_mask_list.append(act_mask)

            unit_accept_cmd = list(head0_pointers)

            # 转换为仿真环境接受的指令
            for num in range(red_fighter_num):
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                if num in unit_accept_cmd:
                    if head1_act[0] == 0:
                        true_action[0] = int((360/COURSE_NUM) * head2_act)
                    if head1_act[0] == 1:
                        true_action[3] = head2_act
                red_fighter_action.append(true_action)

            red_fighter_action = np.array(red_fighter_action)

            # step and get next obs
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
            next_red_obs_dict, next_blue_obs_dict = env.get_obs()
            next_fighter_tmp_obs = next_red_obs_dict['fighter']
            next_enemy_tmp_obs = next_red_obs_dict['enemy']
            next_obs_list.append(next_fighter_tmp_obs)
            next_en_obs_list.append(next_enemy_tmp_obs)
            
            # calculate reward 
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
            detector_reward = red_detector_reward
            fighter_reward = red_fighter_reward
            reward_list.append(sum(fighter_reward))

            done = env.get_done()
            step_cnt += 1 
            print('step: ', step_cnt)

            if done:
                t = {}  # 暂时没想好什么处理
                break
            elif len(obs_list) >= BATCH_SIZE:
                # print('start train')
                fighter_model.learn(obs_list, en_obs_list, next_obs_list, next_en_obs_list, memory_mask1_list, memory_mask2_list, action_mask_list,
                reward_list, action_head0_list, action_head1_list, action_head2_list, probs_head0_list, logits_head1_list, probs_head2_list, behavior_value_list)
                train_epoch += 1
                time_to_save = True
                
                # summ = fighter_model.output_summ()
                # fighter_model.train_writer.add_summary(summ, step_cnt)

                obs_list.clear()
                en_obs_list.clear() 
                next_obs_list.clear() 
                next_en_obs_list.clear()
                memory_mask1_list.clear() 
                memory_mask2_list.clear() 
                action_mask_list.clear()
                reward_list.clear() 
                action_head0_list.clear() 
                action_head1_list.clear() 
                action_head2_list.clear() 
                probs_head0_list.clear() 
                logits_head1_list.clear() 
                probs_head2_list.clear() 
                behavior_value_list.clear()
            
            # save model.
            # if (train_epoch > 0) and (train_epoch % SAVE_MODEL_INTERVAL == 0) and time_to_save:
            #     print('saving model !')
            #     fighter_model.save_model(model_path, model_name, str(train_epoch))
            #     time_to_save = False

            