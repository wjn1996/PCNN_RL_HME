import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random as rd

# 环境
# 环境不包含任何模型，只用来处理给定的一个动作，获取其对应的下一个状态
class Environment():

    def __init__(self,sentence_len):
        self.sentence_len = sentence_len # 句子向量长度

    # 恢复当前包的初始状态
    def reset(self, h_t, batch_sentence_ebd):
        self.h_t = h_t # 头尾实体的差（t-h）
        self.batch_len = len(batch_sentence_ebd) # 包内句子的个数
        self.sentence_ebd = batch_sentence_ebd
        self.current_step = 0
        self.num_selected = 0
        self.list_selected = []
        self.list_noise = []
        self.vector_current = self.sentence_ebd[self.current_step]
        self.vector_mean = np.zeros(self.sentence_len, dtype=np.float32)
        self.vector_sum = np.zeros(self.sentence_len, dtype=np.float32)

        current_state = np.concatenate([self.vector_current, self.vector_mean, self.h_t])
        return current_state

    # 执行一个动作后，刷新状态
    def step(self, action):

        if action == 1:
            self.num_selected += 1
            self.list_selected.append(self.current_step)
        else :
            self.list_noise.append(self.current_step)
        # print('self.action=', action)
        # print('self.vector_current=', self.vector_current)
        # print('self.vector_sum=', self.vector_sum)
        self.vector_sum = self.vector_sum + action * np.array(self.vector_current)
        if self.num_selected == 0:
            self.vector_mean = np.array([0.0 for x in range(self.sentence_len)], dtype=np.float32)
        else:
            self.vector_mean = self.vector_sum / self.num_selected

        self.current_step += 1

        if self.current_step < self.batch_len:
            self.vector_current = self.sentence_ebd[self.current_step]

        current_state = np.concatenate([self.vector_current, self.vector_mean, self.h_t])
        return current_state

    def reward(self, tag):
        assert (len(self.list_selected) == self.num_selected)
        select_weight, noise_weight = 0., 0.
        if self.num_selected > 0:
            select_weight = self.num_selected/self.batch_len
        if len(self.list_selected) > 0:
            noise_weight = 1 - select_weight
        n_s = 0
        n_n = 0
        for i in self.list_selected:
            if tag[i] == 1:
                n_s += 1
        for i in self.list_noise:
            if tag[i] == 0:
                n_n += 1
        if len(self.list_selected) != 0 and len(self.list_noise) != 0:
            reward =  select_weight * (n_s/len(self.list_selected)) + noise_weight * (n_n/len(self.list_noise))
        elif len(self.list_selected) == 0 and len(self.list_noise) != 0:
            reward =  noise_weight * (n_n/len(self.list_noise))
        elif len(self.list_selected) != 0 and len(self.list_noise) == 0:
            reward =  select_weight * (n_s/len(self.list_selected))
        else:
            reward = 0.
        return reward


