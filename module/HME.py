import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
# from configure import args
from load_data import find_node

# Hierarchical Memory Extractor 分层记忆抽取器
class HME(nn.Module):
    '''
    分层记忆抽取器：输入一个bag embedding，和分层关系，输出对应的标签
    
    '''

    def __init__(self, args, rel_embs, hie_rel_tot, hie_rel):
        super(HME, self).__init__()
        self.args = args
        self.model_name = 'Hierarchical Memory Extractor'
        self.rel_embs = rel_embs # 分层标签 [[...], [...]]
        self.cell = Variable(torch.randn(hie_rel_tot + 1, self.args.hie_dim), requires_grad=False)
        self.hie_rel = hie_rel

        # 信息融合：通过一个MLP将bag embedding与实体对t-h进行融合
        self.fusion = nn.Linear(self.args.sent_dim + self.args.ent_dim, self.args.hie_dim)
        # 输入门：决定多少信息从包向量中提取出并保存至cell中
        self.input_gate_h1 = nn.Linear(self.args.sent_dim + self.args.hie_dim, 1)
        self.input_gate_h2 = nn.Linear(self.args.sent_dim + self.args.hie_dim, 1)
        self.input_gate_h3 = nn.Linear(self.args.sent_dim + self.args.hie_dim, 1)
        # 输出门：决定有多少信息从cell中读取
        self.output_gate_h1 = nn.Linear(self.args.sent_dim + self.args.hie_dim, 1)
        self.output_gate_h2 = nn.Linear(self.args.sent_dim + self.args.hie_dim, 1)
        self.output_gate_h3 = nn.Linear(self.args.sent_dim + self.args.hie_dim, 1)

        #相似度计算矩阵

        self.W = Variable(torch.randn(self.args.hie_dim, self.args.hie_dim), requires_grad=True)

    def hierarchical_similarity(self, z, r):
        # z表示从cell中和bag中提取的语义向量 [hie_dim]
        # r表示当前层的下一层标签向量 [hie_dim]

        z = z.expand([1, z.shape[0]]) # [1, hie_dim]
        r = r.expand([1, r.shape[0]]) # [1, hie_dim]
        # print('z.shape=', z.shape)
        # print('r.shape=', r.shape)
        # r = r.permute(1, 0)
        # q = torch.mm(torch.mm(z, self.W), r) # [1, hie_dim] * [hie_dim, hie_dim] * [hie_dim, 1] = [n, 1, 1]
        # q = torch.mm(torch.mm(z, self.W), r)
        # q = F.softmax(q)
        q = F.cosine_similarity(z,r)
        q = torch.squeeze(q) # ()
        return q

    def prob_layer(self, i, bag_embedding, layer, node, relation):
        if layer == 1:
            rel_l1, _ = find_node(node, 2, self.hie_rel)
            rel_l1 = list(rel_l1)
            i_h2 = F.sigmoid(self.input_gate_h2(torch.cat([bag_embedding[i], self.cell[relation[i][2]]], -1)))
            new_cell = i_h2 * self.new_emb[i] + (1 - i_h2) * self.cell[relation[i][2]]
            o_h2 = F.sigmoid(self.output_gate_h2(torch.cat([bag_embedding[i], new_cell], -1)))
            z_h2 = o_h2 * self.cell[relation[i][2]] + (1 - o_h2) * self.new_emb[i]
            # 计算layer1层各结点的相似度，并获取相似度最大的
            max_l1 = -1
            max_sim = 0.0
            prob_l1 = []
            for j in rel_l1:
                hs = self.hierarchical_similarity(z_h2, self.rel_embs[j])
                prob_l1.append(hs)
                if hs > max_sim:
                    max_l1 = j
                    max_sim = hs
            # prob_l1 = F.softmax(torch.stack(prob_l1))
            prob_l1 = torch.stack(prob_l1)
            for ej, j in enumerate(rel_l1):
                # self.prob[j] = prob_l1[ej] * self.prob[node]
                self.prob[j] = prob_l1[ej] + self.prob[node]

            for j in rel_l1:
                self.prob_layer(i, bag_embedding, layer=0, node=j, relation=relation)
        elif layer == 0:
            rel_l0, _ = find_node(node, 1, self.hie_rel)
            rel_l0 = list(rel_l0)
            i_h3 = F.sigmoid(self.input_gate_h3(torch.cat([bag_embedding[i], self.cell[relation[i][1]]], -1)))
            new_cell = i_h3 * self.new_emb[i] + (1 - i_h3) * self.cell[relation[i][1]]
            o_h3 = F.sigmoid(self.output_gate_h3(torch.cat([bag_embedding[i], new_cell], -1)))
            z_h3 = o_h3 * self.cell[relation[i][1]] + (1 - o_h3) * self.new_emb[i]
            # 计算layer0层各结点的相似度，并获取相似度最大的
            max_l0 = -1
            max_sim = 0.0
            prob_l0 = []
            for j in rel_l0:
                hs = self.hierarchical_similarity(z_h3, self.rel_embs[j])
                prob_l0.append(hs)
                if hs > max_sim:
                    max_l0 = j
                    max_sim = hs
            # prob_l0 = F.softmax(torch.stack(prob_l0))
            prob_l0 = torch.stack(prob_l0)
            for ej, j in enumerate(rel_l0):
                # self.prob[j] = prob_l0[ej] * self.prob[node]
                self.prob[j] = prob_l0[ej] + self.prob[node]



    def prob_dis(self, bag_embedding, relation, head, tail):
        # 为了能够获得PR曲线和AUC面积，需要获取每个类的概率分布，因此需要对整个关系标签树进行深搜以获得每个类的概率分布
        self.ent_rel = head - tail
        self.new_emb = self.fusion(torch.cat([bag_embedding, self.ent_rel], -1)) # [n, hie_dim]
        rel_pred = []
        probs = []
        # 遍历每一个包
        for i in range(bag_embedding.shape[0]):
            self.prob = torch.zeros(self.rel_embs.shape[0])
            # =========layer2==========
            rel_l2, _ = find_node(-1, 3, self.hie_rel)
            rel_l2 = list(rel_l2)
            i_h1 = F.sigmoid(self.input_gate_h1(torch.cat([bag_embedding[i], self.cell[-1]], -1)))
            new_cell = i_h1 * self.new_emb[i] + (1 - i_h1) * self.cell[-1]
            o_h1 = F.sigmoid(self.output_gate_h1(torch.cat([bag_embedding[i], new_cell], -1)))
            z_h1 = o_h1 * self.cell[-1] + (1 - o_h1) * self.new_emb[i]
            # 计算layer2层各结点的相似度，并获取相似度最大的
            max_l2 = -1
            max_sim = 0.0
            prob_l2 = []
            for j in rel_l2:
                hs = self.hierarchical_similarity(z_h1, self.rel_embs[j])
                prob_l2.append(hs)
                if hs > max_sim:
                    max_l2 = j
                    max_sim = hs
            # prob_l2 = F.softmax(torch.stack(prob_l2))
            prob_l2 = torch.stack(prob_l2)
            for ej, j in enumerate(rel_l2):
                self.prob[j] = prob_l2[ej]

            # 递归计算各层的概率分布
            for j in rel_l2:
                self.prob_layer(i, bag_embedding, layer=1, node=j, relation=relation)
            max_l0 = torch.argmax(self.prob[:53])
            max_l1 = torch.argmax(self.prob[53:89])
            max_l2 = torch.argmax(self.prob[89:])
            self.prob = torch.cat([F.softmax(self.prob[:53]),F.softmax(self.prob[53:89]),F.softmax(self.prob[89:])])
            probs.append(self.prob)
            rel_pred.append([max_l0, max_l1, max_l2])
        probs = torch.stack(probs) # 每个包对应的概率分布（一个向量包含你三层的概率分布，分段后表示即可）
        return probs, rel_pred


    def forward(self, bag_embedding, relation, head, tail, train=True):
        # bag_embedding :[n, sent_dim]
        # relation: [n,3]，每个包对应的三层relation
        # 分层记忆细胞 [n, hie_dim] 与rel_emb结构一样
        # 每次训练时动态更新这个cell，并加载最新的cell
        # train:True时，表示训练过程中，此时需要知道当前包对应的正确标签；False时，表示测试过程，此时根据每一层计算相似度进行搜索
        # return: loss, predict relation, cell
        
        self.ent_rel = head - tail
        self.new_emb = self.fusion(torch.cat([bag_embedding, self.ent_rel], -1)) # [n, hie_dim]
        if train == True:
            # 自顶向下的进行相似度计算，一共是3层，自顶向下分别为layer3（root），layer2，layer1和layer0，其中layer0不需要再计算了
            # 从根结点开始，返回layer2的所有结点的集合            
            # loss = torch.zeros(1)[0]
            loss = 0.0
            # 遍历每一个包
            for i in range(bag_embedding.shape[0]):
                # ======生成layer2语义向量======
                rel_l2, _ = find_node(-1, 3, self.hie_rel)
                u_2 = len(rel_l2) # 用于分配每一层的loss权重，动态权重分配
                i_h1 = F.sigmoid(self.input_gate_h1(torch.cat([bag_embedding[i], self.cell[-1]], -1)))
                new_cell_h1 = i_h1 * self.new_emb[i] + (1 - i_h1) * self.cell[-1]
                # self.cell[-1] = new_cell
                o_h1 = F.sigmoid(self.output_gate_h1(torch.cat([bag_embedding[i], self.cell[-1]], -1)))
                z_h1 = o_h1 * self.cell[-1] + (1 - o_h1) * self.new_emb[i]
                # =======计算layer2的相似度========
                # 从集合中去掉当前包对应的relation id
                rel_l2.remove(relation[i][2])
                rel_l2 = list(rel_l2)
                # 随机从该集合中抽取两个个作为负样本(如果将所有的负样本进行计算可能加大计算量和时间，因此使用随机采样方法)
                neg1 = random.randint(0, len(rel_l2) - 1)
                neg2 = random.randint(0, len(rel_l2) - 1)
                neg_rel_emb1, neg_rel_emb2 = self.rel_embs[rel_l2[neg1]], self.rel_embs[rel_l2[neg1]]
                l1, l2 = self.hierarchical_similarity(z_h1, neg_rel_emb1), self.hierarchical_similarity(z_h1, neg_rel_emb2)
                l3 = self.hierarchical_similarity(z_h1, self.rel_embs[relation[i][2]])
                loss_2 = max(0, l1 + self.args.gamma - l3) + max(0, l2 + self.args.gamma - l3)
                # loss_2 = l1 + self.args.gamma - l3 + l2 + self.args.gamma - l3

                # ======生成layer1语义向量======
                rel_l1, rel_l1_other = find_node(relation[i][2], 2, self.hie_rel)
                u_1 = len(rel_l1)
                i_h2 = F.sigmoid(self.input_gate_h2(torch.cat([bag_embedding[i], self.cell[relation[i][2]]], -1)))
                new_cell_h2 = i_h2 * self.new_emb[i] + (1 - i_h2) * self.cell[relation[i][2]]
                # self.cell[relation[i][2]] = new_cell
                o_h2 = F.sigmoid(self.output_gate_h2(torch.cat([bag_embedding[i], self.cell[relation[i][2]]], -1)))
                z_h2 = o_h2 * self.cell[relation[i][2]] + (1 - o_h2) * self.new_emb[i]
                # =======计算layer1的相似度========
                rel_l1.remove(relation[i][1])
                rel_l1 = list(rel_l1)
                rel_l1_other = list(rel_l1_other)
                if len(rel_l1) > 0:
                    neg1 = random.randint(0, len(rel_l1) - 1)
                    neg2 = random.randint(0, len(rel_l1_other) - 1)
                    neg_rel_emb1, neg_rel_emb2 = self.rel_embs[rel_l1[neg1]], self.rel_embs[rel_l1_other[neg2]]
                    l1, l2 = self.hierarchical_similarity(z_h2, neg_rel_emb1), self.hierarchical_similarity(z_h2, neg_rel_emb2)
                else:
                    neg1 = random.randint(0, len(rel_l1_other) - 1)
                    neg2 = random.randint(0, len(rel_l1_other) - 1)
                    neg_rel_emb1, neg_rel_emb2 = self.rel_embs[rel_l1_other[neg1]], self.rel_embs[rel_l1_other[neg2]]
                    l1, l2 = self.hierarchical_similarity(z_h2, neg_rel_emb1), self.hierarchical_similarity(z_h2, neg_rel_emb2)
                l3 = self.hierarchical_similarity(z_h2, self.rel_embs[relation[i][1]])
                loss_1 = max(0,  l1 + self.args.gamma - l3) + max(0,  l2 + self.args.gamma - l3)
                # loss_1 = l1 + self.args.gamma - l3 + l2 + self.args.gamma - l3

                # ======生成layer0语义向量======
                rel_l0, rel_l0_other = find_node(relation[i][1], 1, self.hie_rel)
                u_0 = len(rel_l0)
                i_h3 = F.sigmoid(self.input_gate_h3(torch.cat([bag_embedding[i], self.cell[relation[i][1]]], -1)))
                new_cell_h3 = i_h3 * self.new_emb[i] + (1 - i_h3) * self.cell[relation[i][1]]
                # self.cell[relation[i][1]] = new_cell
                o_h3 = F.sigmoid(self.output_gate_h3(torch.cat([bag_embedding[i], self.cell[relation[i][1]]], -1)))
                z_h3 = o_h3 * self.cell[relation[i][1]] + (1 - o_h3) * self.new_emb[i]
                # =======计算layer0的相似度========
                rel_l0.remove(relation[i][0])
                rel_l0 = list(rel_l0)
                rel_l0_other = list(rel_l0_other)
                if len(rel_l0) > 0:
                    neg1 = random.randint(0, len(rel_l0) - 1)
                    neg2 = random.randint(0, len(rel_l0_other) - 1)
                    neg_rel_emb1, neg_rel_emb2 = self.rel_embs[rel_l0[neg1]], self.rel_embs[rel_l0_other[neg2]]
                    l1, l2 = self.hierarchical_similarity(z_h3, neg_rel_emb1), self.hierarchical_similarity(z_h3, neg_rel_emb2)
                else:
                    neg1 = random.randint(0, len(rel_l0_other) - 1)
                    neg2 = random.randint(0, len(rel_l0_other) - 1)
                    neg_rel_emb1, neg_rel_emb2 = self.rel_embs[rel_l0_other[neg1]], self.rel_embs[rel_l0_other[neg2]]
                    l1, l2 = self.hierarchical_similarity(z_h3, neg_rel_emb1), self.hierarchical_similarity(z_h3, neg_rel_emb2)
                l3 = self.hierarchical_similarity(z_h3, self.rel_embs[relation[i][0]])
                loss_0 = max(0,  l1 + self.args.gamma - l3) + max(0,  l2 + self.args.gamma - l3)
                # loss_0 = l1 + self.args.gamma - l3 + l2 + self.args.gamma - l3
                #将三层的loss进行加权求和
                loss = loss + u_0/(u_0+u_1+u_2+3) * loss_0 + (u_1+1)/(u_0+u_1+u_2+3) * loss_1 + (u_1+2)/(u_0+u_1+u_2+3) * loss_2
                # self.cell[-1] = new_cell_h1.detach()
                # self.cell[relation[i][2]] = new_cell_h2.detach()
                # self.cell[relation[i][1]] = new_cell_h3.detach()
            # 一个batch内所有包对应的loss求平均
            loss = loss/bag_embedding.shape[0]
            # print('loss=', loss)
            return loss, _, self.cell
        else:
            rel_pred = []
            # 遍历每一个包
            for i in range(bag_embedding.shape[0]):
                # =========layer2==========
                rel_l2, _ = find_node(-1, 3, self.hie_rel)
                rel_l2 = list(rel_l2)
                i_h1 = F.sigmoid(self.input_gate_h1(torch.cat([bag_embedding[i], self.cell[-1]], -1)))
                new_cell = i_h1 * self.new_emb[i] + (1 - i_h1) * self.cell[-1]
                o_h1 = F.sigmoid(self.output_gate_h1(torch.cat([bag_embedding[i], new_cell], -1)))
                z_h1 = o_h1 * self.cell[-1] + (1 - o_h1) * self.new_emb[i]
                # 计算layer2层各结点的相似度，并获取相似度最大的
                max_l2 = -1
                max_sim = 0.0
                for j in rel_l2:
                    hs = self.hierarchical_similarity(z_h1, self.rel_embs[j])
                    if hs > max_sim:
                        max_l2 = j
                        max_sim = hs
                # =========layer1==========
                rel_l1, _ = find_node(max_l2, 2, self.hie_rel)
                rel_l1 = list(rel_l1)
                i_h2 = F.sigmoid(self.input_gate_h2(torch.cat([bag_embedding[i], self.cell[relation[i][2]]], -1)))
                new_cell = i_h2 * self.new_emb[i] + (1 - i_h2) * self.cell[relation[i][2]]
                o_h2 = F.sigmoid(self.output_gate_h2(torch.cat([bag_embedding[i], new_cell], -1)))
                z_h2 = o_h2 * self.cell[relation[i][2]] + (1 - o_h2) * self.new_emb[i]
                # 计算layer1层各结点的相似度，并获取相似度最大的
                max_l1 = -1
                max_sim = 0.0
                for j in rel_l1:
                    hs = self.hierarchical_similarity(z_h2, self.rel_embs[j])
                    if hs > max_sim:
                        max_l1 = j
                        max_sim = hs
                rel_l0, _ = find_node(max_l1, 1, self.hie_rel)
                rel_l0 = list(rel_l0)
                i_h3 = F.sigmoid(self.input_gate_h3(torch.cat([bag_embedding[i], self.cell[relation[i][1]]], -1)))
                new_cell = i_h3 * self.new_emb[i] + (1 - i_h3) * self.cell[relation[i][1]]
                o_h3 = F.sigmoid(self.output_gate_h3(torch.cat([bag_embedding[i], new_cell], -1)))
                z_h3 = o_h3 * self.cell[relation[i][1]] + (1 - o_h3) * self.new_emb[i]
                # 计算layer0层各结点的相似度，并获取相似度最大的
                max_l0 = -1
                max_sim = 0.0
                for j in rel_l0:
                    hs = self.hierarchical_similarity(z_h3, self.rel_embs[j])
                    if hs > max_sim:
                        max_l0 = j
                        max_sim = hs
                rel_pred.append([max_l0, max_l1, max_l2])
            return torch.Tensor([0])[0], rel_pred, self.cell