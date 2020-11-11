import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random as rd

# 智能体
# 根据当前的状态，执行相应的动作
class InstanceDetector(nn.Module):
    def __init__(self, args):
        super(InstanceDetector, self).__init__()
        self.args = args
        self.state_dim = self.args.sent_dim*2 + self.args.ent_dim
        self.Linear = nn.Linear(self.state_dim, 1)
        self.optim = optim.SGD(self.parameters(), lr = 0.01)

        self.state = []
        self.action = []
        self.reward = [] #torch.empty(0)

    # 保存采样得到的一幕序列（一个包内的每个时刻的状态-动作及奖励三元组）
    def store_episode(self, state, action, reward):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
    # 重置
    def resume_episode(self):
        self.state, self.action, self.reward = [], [], []

    # 训练时不使用：根据当前的动作概率，决定采样（带有贪心策略），返回贪心后执行的动作
    def opt_action(self, prob):
        return 0 if prob<0.5 else 1

    def select_action(self, prob, is_epsilon=True):      
        if is_epsilon:
            # epsilon贪心选择动作：有epsilon的概率选择最优动作，剩下概率随机选择一个动作
            # seed = rd.random()*100
            # if seed < self.args.epsilon*100:
            #     return self.opt_action(prob)
            # else:                
            #     return self.opt_action(rd.random())

            result = np.random.rand()
            if result>0 and result< prob:
                return 1
            elif result >=prob and result<1:
                return 0 
        else:
            return self.opt_action(prob)
    # def sample(self):
    #     with torch.no_grad():
    #         for i in range(self.args.num_mc_sample):


    # 根据采样的当前一幕，进行训练
    def train(self):
        state = torch.Tensor(self.state) # [n, 2*sent_dim+ent_dim]
        action = torch.Tensor(self.action).unsqueeze(1) # [n]
        reward = torch.Tensor(self.reward).unsqueeze(1) # [n]
        # print(reward)
        # print('state.shape=', state.shape)
        prob = self.forward(state) # [n]
        self.pi = action*prob + (1-action)*(1-prob)
        # print('pi=', self.pi)
        self.loss = torch.sum(-1 * torch.log(self.pi) * reward)
        self.optim.zero_grad() 
        self.loss.backward(retain_graph=True)
        self.optim.step()
        
    
    # 根据训练好的策略，对当前的包进行挑选句子
    def test(self, state):
        with torch.no_grad():
            prob = self.forward(state)
            action = self.select_action(prob, is_epsilon=False)

    def forward(self, state):
        prob = F.sigmoid(self.Linear(state))
        return prob


