import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random as rd
from module.HME import HME
# from configure import args


class Encoder_PCNN(nn.Module):
    '''
    Zeng 2015 DS PCNN
    '''
    def __init__(self, args, vocab_size, pos_size, rel_tot, hie_rel_tot, vec, rel_emb, hie_rel, use_pcnn=True):
        super(Encoder_PCNN, self).__init__()
        self.args = args
        self.model_name = 'PCNN + HME'
        self.vocab_size = vocab_size # 词表大小
        self.pos_size = pos_size # 位置范围
        self.rel_tot = rel_tot # 关系数量
        self.hie_rel_tot = hie_rel_tot # 分层关系总数
        self.vec = vec # 预训练词向量表
        self.hie_rel = hie_rel
        self.rel_emb = rel_emb # 预训练分层标签向量
        self.use_pcnn = use_pcnn
        self.word_embs = nn.Embedding(self.vocab_size, self.args.word_dim)
        self.pos1_embs = nn.Embedding(self.pos_size, self.args.pos_dim)
        self.pos2_embs = nn.Embedding(self.pos_size, self.args.pos_dim)
        self.rel_embs = Variable(torch.Tensor(self.rel_emb), requires_grad=False)
        # self.rel_embs = nn.Embedding(self.hie_rel_tot, self.args.rel_dim)
        self.cell_emb = Variable(torch.randn(self.hie_rel_tot, self.args.hie_dim), requires_grad=False) # 初始化共享信息
        # self.cell_emb = nn.Embedding(self.hie_rel_tot, self.args.hie_dim)
        # self.cell_emb.weight.data.copy_(self.cell)

        feature_dim = self.args.word_dim + self.args.pos_dim * 2

        # for more filter size
        self.convs = nn.ModuleList([nn.Conv2d(1, self.args.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in self.args.filters])

        all_filter_num = self.args.filters_num * len(self.args.filters)

        if self.use_pcnn:
            all_filter_num = all_filter_num * 3
            masks = torch.FloatTensor(([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            if self.args.use_gpu:
                masks = masks.cuda()
            self.mask_embedding = nn.Embedding(4, 3)
            self.mask_embedding.weight.data.copy_(masks)
            self.mask_embedding.weight.requires_grad = False

        self.linear = nn.Linear(all_filter_num, self.args.sent_dim)
        self.sent_linear = nn.Linear(self.args.sent_dim, self.rel_tot)
        # self.FFN = nn.Linear(self.args.sent_dim + self.args.rel_dim, self.args.hie_dim)
        self.HME = HME(self.args, self.rel_embs, self.hie_rel_tot, self.hie_rel)
        self.dropout = nn.Dropout(self.args.drop_out)
        self.init_word_emb(self.vec, self.rel_emb)
        self.init_model_weight()
        self.optim = optim.Adam(self.parameters())
        self.criterion  = nn.CrossEntropyLoss()
        

    def init_model_weight(self):
        '''
        use xavier to init
        '''
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0.0)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def init_word_emb(self, vec, rel_emb):

        w2v = torch.from_numpy(vec)
        rel2v = torch.from_numpy(rel_emb)

        # w2v = torch.div(w2v, w2v.norm(2, 1).unsqueeze(1))
        # w2v[w2v != w2v] = 0.0

        # if self.args.use_gpu:
        #     self.word_embs.weight.data.copy_(w2v.cuda())
        #     self.rel_embs.weight.data.copy_(rel2v.cuda())
        # else:
        # self.word_embs.weight.data.copy_(w2v)
        # self.rel_embs.weight.data.copy_(rel2v)

    def mask_piece_pooling(self, x, mask):
        '''
        refer: https://github.com/thunlp/OpenNRE
        A fast piecewise pooling using mask
        '''
        x = x.unsqueeze(-1).permute(0, 2, 1, -1)
        masks = self.mask_embedding(mask.long()).unsqueeze(-2) * 100
        x = masks.float() + x
        if self.args.use_gpu:
            x = torch.max(x, 1)[0] - torch.FloatTensor([100]).cuda()
        else:
            x = torch.max(x, 1)[0] - torch.FloatTensor([100])
        x = x.view(-1, x.size(1) * x.size(2))
        return x

    def piece_max_pooling(self, x, insPool):
        '''
        old version piecewise
        '''
        split_batch_x = torch.split(x, 1, 0)
        split_pool = torch.split(insPool, 1, 0)
        batch_res = []
        for i in range(len(split_pool)):
            ins = split_batch_x[i].squeeze()  # all_filter_num * max_len
            pool = split_pool[i].squeeze().data    # 2
            seg_1 = ins[:, :pool[0]].max(1)[0].unsqueeze(1)          # all_filter_num * 1
            seg_2 = ins[:, pool[0]: pool[1]].max(1)[0].unsqueeze(1)  # all_filter_num * 1
            seg_3 = ins[:, pool[1]:].max(1)[0].unsqueeze(1)
            piece_max_pool = torch.cat([seg_1, seg_2, seg_3], 1).view(1, -1)    # 1 * 3all_filter_num
            batch_res.append(piece_max_pool)

        out = torch.cat(batch_res, 0)
        assert out.size(1) == 3 * self.args.filters_num
        return out

    # # 分层共享Embedding
    # def hierachical_share(self, bag_embedding, hie_rel, train=False):
    #     # bag_embedding: [batch_size, sent_dim]
    #     # hie_rel:[batch_size, 3]
    #     alpha = 0.9
    #     if train:
    #         rel_emb = self.rel_embs(hie_rel)
            
    #         rel_emb_h1 = rel_emb[:][0]
    #         rel_emb_h2 = rel_emb[:][1]
    #         rel_emb_h3 = rel_emb[:][2]
    #         b_r_h1 = torch.cat([bag_embedding, rel_emb_h1], -1)
    #         b_r_h2 = torch.cat([bag_embedding, rel_emb_h2], -1)
    #         b_r_h3 = torch.cat([bag_embedding, rel_emb_h3], -1)
    #         new_cell_h1 = F.tanh(self.FFN(b_r_h1))
    #         new_cell_h2 = F.tanh(self.FFN(b_r_h2))
    #         new_cell_h3 = F.tanh(self.FFN(b_r_h3))

    #         cell_h1 = []
    #         cell_h2 = []
    #         cell_h3 = []
    #         # 遍历每一个包
    #         for i in range(bag_embedding.shape[0]):
    #             cell_emb = self.cell_emb(hie_rel[i])
    #             ch1 = alpha * new_cell_h1[i] + (1 - alpha) * cell_emb[i][0]
    #             ch2 = alpha * new_cell_h2[i] + (1 - alpha) * cell_emb[i][1]
    #             ch3 = alpha * new_cell_h3[i] + (1 - alpha) * cell_emb[i][2]
    #             cell_h1.append(ch1)
    #             cell_h2.append(ch2)
    #             cell_h3.append(ch3)
    #             self.cell[hie_rel[0]] = ch1
    #             self.cell[hie_rel[1]] = ch2
    #             self.cell[hie_rel[2]] = ch3
    #             self.cell_emb.weight.data.copy_(self.cell)
    #         cell_h1 = torch.stack(cell_h1)
    #         cell_h2 = torch.stack(cell_h2)
    #         cell_h3 = torch.stack(cell_h3)
    #         hie_embedding = torch.cat([cell_h1, cell_h2, cell_h3], -1)
    #     # else:


    # def update_cell(self, new_cell):
    #     self.cell = new_cell


    # # 仅在对策略进行蒙特卡洛采样时使用该函数获得reward
    # def BagExtractor(self, bag_embedding):
    #     # bag_embedding: [batch_size,sent_dim]
    #     y_hat = F.softmax(self.bag_linear(bag_embedding))

    #     return y_hat


    def forward(self, x, hie_rel, scope, sent_label, head, tail, train=False, hme_pro=False):
        # x tensor
        # word,pos: [n, max_len]
        # scope:每个包的范围 [batch_size, 2]
        # hie_rel tensor:[batch_size, 3]
        # head：每个batch对应的头实体
        # tail：每个batch对应的尾实体
        word, pos1, pos2, insMasks = x
        # insPF1, insPF2 = [i.squeeze(1) for i in torch.split(insPFs, 1, 1)]

        word_emb = self.word_embs(word.long())
        pos1_emb = self.pos1_embs(pos1.long())
        pos2_emb = self.pos2_embs(pos2.long())
        head_emb = self.word_embs(head.long())
        tail_emb = self.word_embs(tail.long())

        x = torch.cat([word_emb, pos1_emb, pos2_emb], 2)
        x = x.unsqueeze(1)
        x = self.dropout(x)

        x = [conv(x).squeeze(3) for conv in self.convs]
        if self.use_pcnn:
            x = [self.mask_piece_pooling(i, insMasks) for i in x]
            # x = [self.piece_max_pooling(i, insPool) for i in x]
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1).tanh()
        sent_embedding = self.linear(self.dropout(x)) # PCNN获得每个句子的向量
        y_pcnn_pred = self.sent_linear(sent_embedding) # PCNN句子级别进行分类
        loss1 = self.criterion(y_pcnn_pred, sent_label.long())
        # 打包
        bag_embedding = [] # [batch_size, sent_dim]
        for start, end in scope:
            bag_embedding.append(torch.mean(sent_embedding[start: end], 0))
        bag_embedding = torch.stack(bag_embedding)
        # Bag-level Embedding
        # y_bag_hat = self.BagExtractor(bag_embedding)

        # Hierarchical Memory Extractor
        self.HME.cell = self.cell_emb
        if hme_pro == False: # 不需要获取HME的概率分布
            loss2, y_hme_rel, cell = self.HME(bag_embedding, hie_rel, head_emb, tail_emb, train=train)
            self.cell_emb = cell
            self.ent_rel = self.HME.ent_rel
            # y_pcnn_pred:每个关系的概率分布
            # y_hme_rel:三层关系
            return y_pcnn_pred, y_hme_rel, sent_embedding, loss1 + loss2
        else:
            hme_probs, y_hme_rel = self.HME.prob_dis(bag_embedding, hie_rel, head_emb, tail_emb)
            return hme_probs, y_hme_rel



# Encoder
class Encoder_CNN(nn.Module):
    '''
    the basic model
    Zeng 2014 "Relation Classification via Convolutional Deep Neural Network"
    '''

    def __init__(self, args, vocab_size, pos_size, rel_tot, vec):
        super(Encoder_PCNN, self).__init__()
        self.args = args
        self.model_name = 'CNN'
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.rel_tot = rel_tot
        self.vec = vec
        self.word_embs = nn.Embedding(self.vocab_size, self.args.word_dim)
        self.pos1_embs = nn.Embedding(self.pos_size + 1, self.args.pos_dim)
        self.pos2_embs = nn.Embedding(self.pos_size + 1, self.args.pos_dim)

        feature_dim = self.word_dim + self.pos_dim * 2

        # encoding sentence level feature via cnn
        self.convs = nn.ModuleList([nn.Conv2d(1, self.args.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in self.args.filters])
        all_filter_num = self.args.filters_num * len(self.args.filters)
        self.cnn_linear = nn.Linear(all_filter_num, self.args.sent_dim)
        # self.cnn_linear = nn.Linear(all_filter_num, self.opt.rel_num)

        # concat the lexical feature in the out architecture
        self.out_linear = nn.Linear(self.args.sent_dim, self.rel_tot)
        # self.out_linear = nn.Linear(self.opt.sen_feature_dim, self.opt.rel_num)
        self.dropout = nn.Dropout(self.args.drop_out)
        self.init_word_emb(self.vec)
        self.init_model_weight()

    def init_model_weight(self):
        '''
        use xavier to init
        '''
        nn.init.xavier_normal_(self.cnn_linear.weight)
        nn.init.constant_(self.cnn_linear.bias, 0.)
        nn.init.xavier_normal_(self.out_linear.weight)
        nn.init.constant_(self.out_linear.bias, 0.)
        for conv in self.convs:
            nn.init.xavier_normal_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def init_word_emb(self, vec):

        w2v = torch.from_numpy(vec)

        # w2v = torch.div(w2v, w2v.norm(2, 1).unsqueeze(1))
        # w2v[w2v != w2v] = 0.0

        if self.args.use_gpu:
            self.word_embs.weight.data.copy_(w2v.cuda())
        else:
            self.word_embs.weight.data.copy_(w2v)

    def forward(self, x):

        word, pos1, pos2 = x

        # sentence level feature
        word_emb = self.word_embs(word)  # (batch_size, max_len, word_dim)
        pos1_emb = self.pos1_embs(pos1)  # (batch_size, max_len, word_dim)
        pos2_emb = self.pos2_embs(pos2)  # (batch_size, max_len, word_dim)

        sentence_feature = torch.cat([word_emb, pos1_emb, pos2_emb], 2)  # (batch_size, max_len, word_dim + pos_dim *2)

        # conv part
        x = sentence_feature.unsqueeze(1)
        x = self.dropout(x)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        x = self.cnn_linear(x)
        x = self.tanh(x)
        sent_embedding = self.dropout(x)
        y_hat = self.out_linear(sent_embedding)
        return y_hat, sent_embedding

# class Encoder_PCNN(nn.Moudle):
#     '''
#     Zeng 2015 DS PCNN
#     '''
#     def __init__(self, args, vocab_size, pos_size, rel_tot, vec, use_pcnn=True):
#         super(PCNN_ONE, self).__init__()
#         self.args = args
#         self.model_name = 'PCNN'
#         self.vocab_size = vocab_size
#         self.pos_size = pos_size
#         self.rel_tot = rel_tot
#         self.vec = vec
#         self.use_pcnn = use_pcnn

#         self.word_embs = nn.Embedding(self.vocab_size, self.args.word_dim)
#         self.pos1_embs = nn.Embedding(self.pos_size, self.args.pos_dim)
#         self.pos2_embs = nn.Embedding(self.pos_size, self.args.pos_dim)

#         feature_dim = self.args.word_dim + self.args.pos_dim * 2

#         # for more filter size
#         self.convs = nn.ModuleList([nn.Conv2d(1, self.args.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in self.args.filters])

#         all_filter_num = self.args.filters_num * len(self.args.filters)

#         if self.use_pcnn:
#             all_filter_num = all_filter_num * 3
#             masks = torch.FloatTensor(([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
#             if self.args.use_gpu:
#                 masks = masks.cuda()
#             self.mask_embedding = nn.Embedding(4, 3)
#             self.mask_embedding.weight.data.copy_(masks)
#             self.mask_embedding.weight.requires_grad = False

#         self.linear = nn.Linear(all_filter_num, self.rel_tot)
#         self.dropout = nn.Dropout(self.args.drop_out)
#         self.init_word_emb(self.vec)
#         self.init_model_weight()
        

#     def init_model_weight(self):
#         '''
#         use xavier to init
#         '''
#         for conv in self.convs:
#             nn.init.xavier_uniform_(conv.weight)
#             nn.init.constant_(conv.bias, 0.0)

#         nn.init.xavier_uniform_(self.linear.weight)
#         nn.init.constant_(self.linear.bias, 0.0)

#     def init_word_emb(self, vec):

#         w2v = torch.from_numpy(vec)

#         # w2v = torch.div(w2v, w2v.norm(2, 1).unsqueeze(1))
#         # w2v[w2v != w2v] = 0.0

#         if self.args.use_gpu:
#             self.word_embs.weight.data.copy_(w2v.cuda())
#         else:
#             self.word_embs.weight.data.copy_(w2v)

#     def mask_piece_pooling(self, x, mask):
#         '''
#         refer: https://github.com/thunlp/OpenNRE
#         A fast piecewise pooling using mask
#         '''
#         x = x.unsqueeze(-1).permute(0, 2, 1, -1)
#         masks = self.mask_embedding(mask).unsqueeze(-2) * 100
#         x = masks.float() + x
#         x = torch.max(x, 1)[0] - torch.FloatTensor([100]).cuda()
#         x = x.view(-1, x.size(1) * x.size(2))
#         return x

#     def piece_max_pooling(self, x, insPool):
#         '''
#         old version piecewise
#         '''
#         split_batch_x = torch.split(x, 1, 0)
#         split_pool = torch.split(insPool, 1, 0)
#         batch_res = []
#         for i in range(len(split_pool)):
#             ins = split_batch_x[i].squeeze()  # all_filter_num * max_len
#             pool = split_pool[i].squeeze().data    # 2
#             seg_1 = ins[:, :pool[0]].max(1)[0].unsqueeze(1)          # all_filter_num * 1
#             seg_2 = ins[:, pool[0]: pool[1]].max(1)[0].unsqueeze(1)  # all_filter_num * 1
#             seg_3 = ins[:, pool[1]:].max(1)[0].unsqueeze(1)
#             piece_max_pool = torch.cat([seg_1, seg_2, seg_3], 1).view(1, -1)    # 1 * 3all_filter_num
#             batch_res.append(piece_max_pool)

#         out = torch.cat(batch_res, 0)
#         assert out.size(1) == 3 * self.args.filters_num
#         return out       


#     # 仅在对策略进行蒙特卡洛采样时使用该函数获得reward
#     def BagExtractor(self, bag_embedding):
#         # bag_embedding: [1,sent_dim]
#         y_hat = self.linear(bag_embedding)
#         return y_hat


#     def forward(self, x, scope, train=False):
#         # word,pos: [n, max_len]
#         # scope:每个包的范围 [batch_size, 2]
#         word, pos1, pos2, insMasks = x
#         insPF1, insPF2 = [i.squeeze(1) for i in torch.split(insPFs, 1, 1)]

#         word_emb = self.word_embs(word)
#         pos1_emb = self.pos1_embs(pos1)
#         pos2_emb = self.pos2_embs(pos2)

#         x = torch.cat([word_emb, pos1_emb, pos2_emb], 2)
#         x = x.unsqueeze(1)
#         x = self.dropout(x)

#         x = [conv(x).squeeze(3) for conv in self.convs]
#         if self.use_pcnn:
#             x = [self.mask_piece_pooling(i, insMasks) for i in x]
#             # x = [self.piece_max_pooling(i, insPool) for i in x]
#         else:
#             x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
#         x = torch.cat(x, 1).tanh()
#         sent_embedding = self.dropout(x)

#         # 打包
#         bag_embedding = [] # [batch_size, sent_dim]
#         for start, end in scope:
#             bag_embedding.append(sent_embedding[start: end])
#         bag_embedding = torch.stack(bag)

#         y_hat = self.linear(bag_embedding)

#         return y_hat, sent_embedding
