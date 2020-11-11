import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random as rd
from configure import args
import os

# 加载word embedding table
def load_word_embedding_table():
    # # vec : [ 0.198994,  0.219711, -0.190422, -0.162968,  0.067939,  0.150194,
 #        0.046775,  0.010506, -0.179149,  0.110292, -0.216578,  0.062121,
 #       -0.037353, -0.047586, -0.164842, -0.093572,  0.128232,  0.150406,
 #        0.147607,  0.079417,  0.0768  , -0.189049, -0.203621,  0.247066,
 #        0.18898 ,  0.153622, -0.030025,  0.199639, -0.024609,  0.036526,
 #       -0.007419, -0.148312,  0.065239, -0.066491, -0.190179,  0.236354,
 #        0.217716, -0.054444, -0.011242,  0.025314, -0.180848, -0.199214,
 #        0.22644 ,  0.003133, -0.128384, -0.15124 , -0.152947,  0.084363,
 #        0.010013,  0.066172]
    vec = np.load(args.data_path + 'vec.npy')
    print('[load_word_embedding_table] word embedding number:', len(vec))
    print('[load_word_embedding_table] word embedding size:', len(vec[0]))
    return vec


# 处理分层的关系标签
# 默认分为3层
# 返回每个关系标签对应的三层编号，以及所有关系层级标签的向量表
def hierachical_relation():
    def parse(h):
        # /people/person/nationality
        p = h[1:].split('/')
        parent = '/' + '/'.join(p[:-1])
        return parent # # /people/person
    relations = []
    rel2id_h1 = {'NA':0}
    rel2id_h2 = {'NA':0}
    rel2id_h3 = {'NA':0}
    with open('./origin_data/relation2id.txt', encoding='utf-8') as fr:
        relations = fr.readlines()
    for i in relations[1:]:
        name_h1 = i.replace('\n', '').split(' ')[0]
        idx = i.replace('\n', '').split(' ')[1]
        if name_h1 not in rel2id_h1.keys():
            rel2id_h1[name_h1] = int(idx)
        name_h2 = parse(name_h1)
        if name_h2 not in rel2id_h2.keys():
            rel2id_h2[name_h2] = len(rel2id_h2)
        name_h3 = parse(name_h2)
        if name_h3 not in rel2id_h3.keys():
            rel2id_h3[name_h3] = len(rel2id_h3)

    for i in rel2id_h2.keys():
        rel2id_h2[i] = rel2id_h2[i] + len(rel2id_h1)
    for i in rel2id_h3.keys():
        rel2id_h3[i] = rel2id_h3[i] + len(rel2id_h1) + len(rel2id_h2)

    hie_rel = [] # [h1,h2,h3]
    for i in rel2id_h1.keys():
        relid = [0, 0, 0]
        if i != 'NA':
            relid[0] = rel2id_h1[i]
            h2 = parse(i)
            relid[1] = rel2id_h2[h2]
            h3 = parse(h2)
            relid[2] = rel2id_h3[h3]
        else:
            relid = [0, rel2id_h2['NA'], rel2id_h3['NA']]
        hie_rel.append(relid)

    rel_emb = np.random.rand(len(rel2id_h1) + len(rel2id_h2) + len(rel2id_h3), args.rel_dim)

    for i in rel2id_h2.keys():
        num = 0
        em = np.array([0.]*args.rel_dim)
        for j in rel2id_h1.keys():
            if i == parse(j):
                num += 1
                em = em + rel_emb[rel2id_h1[j]]
        em = em/num
        rel_emb[rel2id_h2[i]] = em

    for i in rel2id_h3.keys():
        num = 0
        em = np.array([0.]*args.rel_dim)
        for j in rel2id_h2.keys():
            if i == parse(j):
                num += 1
                em = em + rel_emb[rel2id_h2[j]]
        em = em/num
        rel_emb[rel2id_h3[i]] = em

    if os.path.exists('./model/rel_emb.npy'):
        rel_emb = np.load('./model/rel_emb.npy')
    else:
        np.save('./model/rel_emb.npy', rel_emb)
    
    print('rel2id_h1=', rel2id_h1)
    print('rel2id_h2=', rel2id_h2)
    print('rel2id_h3=', rel2id_h3)
    print('hie_rel=', hie_rel)
    print('rel_emb=', rel_emb)
    # hie_rel:[h1,h2,h3]
    # rel_emb:[99,20]
    print(len(hie_rel))
    print('[load_hierarchical_realtion] hiearachical relation number:', len(rel_emb))
    print('[load_hierarchical_realtion] relation embedding size:', len(rel_emb[0]))
    return hie_rel, rel_emb

def find_node(root, layer, hie_rel):
    # 给定某一层以及父结点，返回其所有的下一层关系标签结点
    # 最底层为layer=0，root虚拟结点为layer=3
    # hie_rel: [h1,h2,h3]
    rels = []
    rels_other = []
    if root == -1 and layer == 3:        
        for i in hie_rel:
            rels.append(i[2])
    elif layer == 2:
        for i in hie_rel:
            if i[2] == root:
                rels.append(i[1])
            else:
                rels_other.append(i[1])
    elif layer == 1:
        for i in hie_rel:
            if i[1] == root:
                rels.append(i[0])
            else:
                rels_other.append(i[0])
    rels = set(rels)
    rels_other = set(rels_other)
    return rels, rels_other




# 加载训练集
def load_train_data():
    # train_X : [[[word1, pos11, pos12 ], [word2, pos21, pos22], ...], [[word1,...]...]]
    # train_y : [[0, 0, ..., 0, 0], [1, 0, 0, ...]]
    # 
    train_word = np.load(args.data_path + 'train_word.npy', allow_pickle=True)
    train_pos1 = np.load(args.data_path + 'train_pos1.npy', allow_pickle=True)
    train_pos2 = np.load(args.data_path + 'train_pos2.npy', allow_pickle=True)
    train_y = np.load(args.data_path + 'train_y.npy', allow_pickle=True)
    train_entity_pair = np.load(args.data_path + 'train_entity_pair.npy', allow_pickle=True)
    print('[load_train_data] training data number:', len(train_y))
    return train_word, train_pos1, train_pos2, train_y, train_entity_pair

# 加载测试集
def load_test_data(scale='testall'):
    '''
    input:(1) scale : 'pone','ptwo','pall','testall', 'less100', 'less200',
    pone:包内示例数大于1， 每个包只挑选一个示例
    ptow:包内示例数大于1，每个包只挑选两个示例
    pall:包内示例数大于1，每个包挑选所有示例
    test_all:保留原始语料
    less100:挑选类标签少于100个包的所有示例
    less200:挑选类标签少于200个包的所有示例
    '''
    filename = 'testall'
    if scale != filename:
        filename = scale + '_test'
    test_word = np.load(args.data_path + filename + '_word.npy', allow_pickle=True)
    test_pos1 = np.load(args.data_path + filename + '_pos1.npy', allow_pickle=True)
    test_pos2 = np.load(args.data_path + filename + '_pos2.npy', allow_pickle=True)
    test_y = np.load(args.data_path + filename + '_y.npy', allow_pickle=True)
    test_entity_pair = np.load(args.data_path + 'test_entity_pair.npy', allow_pickle=True)
    print('[load_test_data(' + scale + ')] test data number:', len(test_y))
    return test_word, test_pos1, test_pos2, test_y, test_entity_pair


# 生成batch
def batch_loader(word, pos1, pos2, y, masks, entity_pair, shuffle=True):
    dataset_size = len(y)
    batch_num = int(dataset_size/args.batch_size) + 1
    if shuffle:
        #随机打乱该数据集顺序（只打乱包的顺序，包内句子顺序未打乱）
        shuffle_indices = np.random.permutation(np.arange(dataset_size))
        word = word[shuffle_indices]
        pos1 = pos1[shuffle_indices]
        pos2 = pos2[shuffle_indices]
        y = y[shuffle_indices]
        masks = masks[shuffle_indices]
        entity_pair = entity_pair[shuffle_indices]
    
    for i in range(batch_num):
        start = i*args.batch_size
        end = (i+1)*args.batch_size
        if end > dataset_size:
            end = dataset_size
        yield i, word[start:end], pos1[start:end], pos2[start:end], y[start:end], masks[start:end], entity_pair[start:end]

# PCNN对每个句子生成mask序列
# 0: [0,0,0] padding部分
# 1: [1,0,0] 头实体左侧部分
# 2: [0,1,0] 头实体以及其与尾实体之间部分
# 3: [0,0,1] 尾实体以及尾实体右侧部分
def load_mask(sentence, entity_pair, sen_len, max_len):
    # print('entity_pair=', entity_pair)
    mask = []
    if entity_pair[0] not in sentence:
        pos1 = sen_len - 2
    else:
        pos1 = sentence.index(entity_pair[0])
    if entity_pair[1] not in sentence:
        pos2 = sen_len - 1
    else:
        pos2 = sentence.index(entity_pair[1])
    mask += [1] * min(pos1, pos2)
    mask += [2] * abs(pos2 - pos1)
    mask += [3] * (sen_len - max(pos1, pos2))
    mask += [0] * (max_len - sen_len)
    return mask

def load_special_token():
    spe = np.load(args.data_path + 'special_token.npy')
    return spe[0], spe[1]

# ========训练模型时辅助的函数==========

# 生成mask
def process_mask(args, word, padding_token, entity_pair):
    masks = []
    for ei, i in enumerate(word):
        mask_ = []
        for j in i:
            if padding_token in j:            
                mask = load_mask(j, entity_pair[ei], j.index(padding_token), args.max_len)
            else:
                mask = load_mask(j, entity_pair[ei], args.max_len-1, args.max_len)
            mask_.append(mask)
        masks.append(mask_)
    return masks

# 处理Tensor
def process_tensor(args, batch_word_, batch_pos1_, batch_pos2_, batch_masks_, batch_entity_pair_, batch_y_, hie_rel):
    batch_word, batch_pos1, batch_pos2, batch_masks, head, tail = [], [], [], [], [], []
    hie_y = [] # 每个batch分层标签（三层，每层对应一个下标）
    y_true = [] # 当前每个batich中正确的标签下标
    scope = [] # 一个batch的范围
    num = 0
    sent_label = [] # 每个句子对应的原始标签下标
    for j in range(len(batch_word_)):
        batch_word += batch_word_[j]
        batch_pos1 += batch_pos1_[j]
        batch_pos2 += batch_pos2_[j]
        batch_masks += batch_masks_[j]
        head += [batch_entity_pair_[j][0]]
        tail += [batch_entity_pair_[j][1]]
        hie_y.append(hie_rel[np.argmax(batch_y_[j])])
        y_true.append(np.argmax(batch_y_[j]))
        scope.append([num, num + len(batch_word_[j])])
        num = num + len(batch_word_[j])
        sent_label += [np.argmax(batch_y_[j])] * len(batch_word_[j])
    x = (torch.Tensor(batch_word), torch.Tensor(batch_pos1), torch.Tensor(batch_pos2), torch.Tensor(batch_masks))
    head = torch.Tensor(head)
    tail = torch.Tensor(tail)
    y_true = torch.Tensor(y_true)
    sent_label = torch.Tensor(sent_label)
    return x, hie_y, scope, sent_label, head, tail, y_true

# 从多个层次上分别判断准确性
def check_res(hie_rel, pred_rel):
    # hie_rel,pred_rel: [h1, h2, h3]
    # 计算预测正确的个数（不计算NA标签的句子）
    # zero = sum(hie_rel[:,0]==0)
    # t = np.array(hie_rel) - np.array(pred_rel)
    # acc_sum_h1 = sum(t[:,0]==0) - zero
    # acc_sum_h2 = sum(t[:,1]==0) - zero
    # acc_sum_h3 = sum(t[:,2]==0) - zero
    acc_sum_h1 = 0
    acc_sum_h2 = 0
    acc_sum_h3 = 0
    test_sum = 0
    for i in range(len(hie_rel)):
        if hie_rel[i][0] != 0:
            test_sum += 1
            if hie_rel[i][0] == pred_rel[i][0]:
                acc_sum_h1 += 1
            if hie_rel[i][1] == pred_rel[i][1]:
                acc_sum_h2 += 1
            if hie_rel[i][2] == pred_rel[i][2]:
                acc_sum_h3 += 1
    return acc_sum_h1, acc_sum_h2, acc_sum_h3, test_sum

# print(args.sent_dim)
# hie_rel, rel_emb = hierachical_relation()
# rels, rels_other = find_node(-1, 3, hie_rel)
# print(rels, rels_other)