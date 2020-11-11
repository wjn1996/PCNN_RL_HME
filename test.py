import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random as rd
from configure import args
from load_data import *
from module.Encoder import Encoder_PCNN
from module.Agent import InstanceDetector
from module.Environment import Environment
from logger import Logger
import time
model_dict = args.model_dict
log = Logger('./', str(int(time.time())))
# 测试
print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] starting load data...')

vec = load_word_embedding_table()
print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] load word embedding success')

test_word, test_pos1, test_pos2, test_y, test_entity_pair = load_test_data()

hie_rel, rel_emb = hierachical_relation()
print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] load hierachical relation success')

padding_token, unknown_token = load_special_token()
pos_size = 200
rel_tot = 53

masks = process_mask(args, test_word, padding_token, test_entity_pair)

# 将预训练的模型重新加载回，并在训练集上对每个句子进行分类，获得每个句子的分类结果，并获得对应的句子表征
print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] evaluate from the test on acc, auc and p@n') 
# net = Encoder_PCNN(args, len(vec), pos_size, rel_tot, len(rel_emb), vec, rel_emb, hie_rel, use_pcnn=args.use_pcnn)
# agent = InstanceDetector(args)

# net.load_state_dict(torch.load('./model/pcnn_hme_pretrain.pkl'))
net = torch.load(model_dict + 'pcnn_hme_pretrain.pkl')
# net.vocab_size = len(vec)
# net.init_word_emb(vec, rel_emb)
# agent.load_state_dict(torch.load('./model/instance_detector_pretrain.pkl'))
agent = torch.load(model_dict + 'instance_detector_pretrain.pkl')
batches = batch_loader(test_word, test_pos1, test_pos2, test_y, np.array(masks), test_entity_pair, shuffle=False)


if args.test_select == True:
    # 测试时，使用智能体挑选句子
    # 首先需要获得测试集每个包内句子的句子向量，并得到每个句子的测试结果
    print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] obtain the sentence-level embeddings and results by trained PCNN\n')
    sent_tag = [] # 保存每个包内每个句子被分类的结果，1表示当前句子分类正确，0表示当前句子分类错误
    sent_embeddings = [] # 保存每个包对应每个句子的表征向量
    t_h = []
    for e, batch_word_, batch_pos1_, batch_pos2_, batch_y_, batch_masks_, batch_entity_pair_ in batches:
        x, hie_y, scope, sent_label, head, tail, y_true = process_tensor(args, batch_word_, batch_pos1_, batch_pos2_, batch_masks_, batch_entity_pair_, batch_y_, hie_rel)
        y_pcnn_pred, _, sent_embedding, _ = net.forward(x, hie_y, scope, sent_label, head, tail, train=False, hme_pro=False)
        ent_rel = net.ent_rel
        y_pred = torch.argmax(y_pcnn_pred, axis=-1)
        # sent_embeddings += sent_embedding.detach().numpy().tolist()
        for start, end in scope:
            sent_embeddings.append(sent_embedding[start: end].detach().numpy().tolist())
            y_true_ = sent_label[start: end]
            y_pred_ = y_pred[start: end]
            sent_tag_ = []
            for i in range(y_true_.shape[0]):
                if y_true_[i] == y_pred_[i]:
                    sent_tag_ += [1]
                else:
                    sent_tag_ += [0]
            sent_tag.append(sent_tag_)
        for i in ent_rel:
            t_h.append(i.detach().numpy().tolist())

    np.save(model_dict + 'test_sentence_embedding.npy', sent_embeddings)
    np.save(model_dict + 'test_sentence_result.npy', sent_tag)
    np.save(model_dict + 'test_t_h.npy', t_h)

    sent_embeddings = np.load(model_dict + 'test_sentence_embedding.npy', allow_pickle=True)
    sent_tag = np.load(model_dict + 'test_sentence_result.npy', allow_pickle=True)
    t_h = np.load(model_dict + 'test_t_h.npy', allow_pickle=True)

    # 现根据已预训练的智能体来对每个包选择句子
    print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] select credible instances (truth) from test data by trained instance detector\n')
    env = Environment(args.sent_dim)
    credible_test_word = []
    credible_test_pos1 = []
    credible_test_pos2 = []
    credible_test_y = []
    credible_test_entity_pair = []
    lost = 0
    for i in range(len(sent_embeddings)):
        batch_sentence_ebd = sent_embeddings[i] # 当前包内每个句子的embedding
        h_t = t_h[i] # 当前包对应的尾实体-头实体向量
        tag = sent_tag[i] # 当前包内每个句子被分类是否正确（正确为1，错误为0）
        # 刷新
        agent.resume_episode()
        current_state = env.reset(h_t, batch_sentence_ebd)
        # step1:蒙特卡洛采样——遍历每一个句子，生成相应的状态
        # print('len(sent_tag[i])=', len(sent_tag[i]), 'len(sent_embeddings[i])=', len(batch_sentence_ebd))
        with torch.no_grad():
            for j in range(len(sent_embeddings[i])):
                prob = agent.forward(torch.Tensor(current_state)) # 根据当前的状态，智能体返回选择/不选择的概率分布
                action = agent.select_action(prob, is_epsilon=False) # 根据概率分布，带有贪心的进行执行动作
                reward = 0 # 非用于训练时，不需要计算reward
                agent.store_episode(current_state, action, reward) # 保存采样序列
                current_state = env.step(action) # 根据当前的动作，环境转移到下一个状态，
        # step2:根据action，将对应当前包内选择（action=1）的句子组合为新的包credible，并保存
        action = agent.action # list当前包的action序列
        slice_index = [ ei for ei, k in enumerate(action) if k == 1]
        if len(slice_index) == 0:# 说明该包内所有句子都未选择，则全部为噪声
            lost += 1
            continue
        # 根据动作，将相应选择的句子组成新包，更新原始的数据集
        a, b, c, = [], [], []
        for j in slice_index:
            a.append(test_word[i][j])
            b.append(test_pos1[i][j])
            c.append(test_pos2[i][j])
        credible_test_word.append(a)
        credible_test_pos1.append(b)
        credible_test_pos2.append(c)
        credible_test_y.append(test_y[i])
        credible_test_entity_pair.append(test_entity_pair[i])
    print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] success for select credible instances\n')
    print('[noisy bag num : ', lost, ', bag num : ', len(sent_embeddings), ']')
    cre_test_word = np.array(credible_test_word)
    cre_test_pos1 = np.array(credible_test_pos1)
    cre_test_pos2 = np.array(credible_test_pos2)
    cre_test_y = np.array(credible_test_y)
    cre_test_entity_pair = np.array(credible_test_entity_pair)

    masks = process_mask(args, cre_test_word, padding_token, cre_test_entity_pair)
    batches = batch_loader(cre_test_word, cre_test_pos1, cre_test_pos2, cre_test_y, np.array(masks), cre_test_entity_pair, shuffle=False)



test_result_l0, test_result_l1, test_result_l2 = [], [], []
probs = []
acc_h1, acc_h2, acc_h3, sum_ = 0, 0, 0, 0
na_NA_num = 0

print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] validate test data\n')
for e, batch_word_, batch_pos1_, batch_pos2_, batch_y_, batch_masks_, batch_entity_pair_ in batches:
    x, hie_y, scope, sent_label, head, tail, y_true = process_tensor(args, batch_word_, batch_pos1_, batch_pos2_, batch_masks_, batch_entity_pair_, batch_y_, hie_rel)
    with torch.no_grad():
        hme_probs, y_hme_rel = net.forward(x, hie_y, scope, sent_label, head, tail, train=False, hme_pro=True)
    probs += hme_probs.detach().numpy().tolist()
    acc_sum_h1, acc_sum_h2, acc_sum_h3, test_sum = check_res(hie_y, y_hme_rel)
    acc_h1 += acc_sum_h1
    acc_h2 += acc_sum_h2
    acc_h3 += acc_sum_h3
    sum_ += test_sum

    # layer0_pro = hme_probs[:,:53]
    # layer1_pro = hme_probs[:,53:89]
    # layer2_pro = hme_probs[:,89:]
    # layer0（原始标签）对应的PR值：
    
    for ei, i in enumerate(hie_y):
        if i[0] != 0:
            na_NA_num += 1
            label_v = [0.]*len(rel_emb)
            label_v[i[0]] = 1.
            label_v[i[1]] = 1.
            label_v[i[2]] = 1.
            for j in range(1, 53): # 0表示NA
                test_result_l0.append({'precision': hme_probs[ei][j], 'label': label_v[j]})
            for j in range(54, 89): # 53表示NA
                test_result_l1.append({'precision': hme_probs[ei][j], 'label': label_v[j]})
            for j in range(90, len(rel_emb)): # 89表示NA
                test_result_l2.append({'precision': hme_probs[ei][j], 'label': label_v[j]})
    # sent_embeddings += sent_embedding.detach().numpy().tolist()
ac_h1 = round(acc_h1*100.0/sum_, 2) # 相当于原始关系的精度（不计算NA的句子）
ac_h2 = round(acc_h2*100.0/sum_, 2) # 只有两层关系的精度（不计算NA的句子）
ac_h3 = round(acc_h3*100.0/sum_, 2) # 只有一层关系的精度（不计算NA的句子） 
np.save(model_dict + 'probs.npy', probs)
log.print_eval(0, test_result_l0, test_result_l1, test_result_l2, na_NA_num, ac_h1, ac_h2, ac_h3)
