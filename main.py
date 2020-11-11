# 联合训练Instance Detector + PCNN + Hierarchical Memory Extractor
'''
    预训练策略：
    step1 :首先执行python3 pretrain.py 对PCNN+HME进行预训练，其次预训练Instance Detector。
    保存预训练的PCNN+HME、Agent模型以及所有当前句子的表征向量
    联合训练：
    step2 :根据当前的智能体（Instance Detector）在整个训练集上每个包进行挑选示例，形成credible和noise两个集合
    step3 :根据挑选的credible集合形成新的训练语料，用来训练PCNN+HME，训练5轮
    step4 :每次训练PCNN+HME时候，都在测试集上进行评估，选择最优的模型
    step5 :根据训练好的PCNN+HME，在训练集上所有示例再次进行测试，得到每个句子分类的结果（0分类错误，1分类正确），以及句子向量
    step6 :Instance Detector进行采样，并进行训练

'''
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

log = Logger('./', str(int(time.time())))
# 预训练PCNN
print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] starting load data...\n')

vec = load_word_embedding_table()
print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] load word embedding success\n')

train_word, train_pos1, train_pos2, train_y, train_entity_pair = load_train_data()
test_word, test_pos1, test_pos2, test_y, test_entity_pair = load_test_data()
print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] load training data success\n')
# print('train_word=', train_word[0])
# print('train_pos1=', train_pos1[0])
# print('train_pos2=', train_pos2[0])
# print('train_y=', train_y[0])
# print('train_entity_pair=', train_entity_pair[0])
hie_rel, rel_emb = hierachical_relation()
print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] load hierachical relation success\n')

padding_token, unknown_token = load_special_token()
pos_size = 200
rel_tot = 53


# 创建一个PCNN模型对象
beta = 0.5
max_acc = 0.0
if os.path.exists('./model/best_acc.npy'):
    max_acc = np.load('./model/best_acc.npy')
print('pretrain max acc=', max_acc)
cotrain_step = 0
# ========联合训练Instance Detector和PCNN+HME===========

# net = Encoder_PCNN(args, len(vec), pos_size, rel_tot, len(rel_emb), vec, rel_emb, hie_rel, use_pcnn=args.use_pcnn)
# agent = InstanceDetector(args)

for epoch in range(1, args.num_epochs + 1):

    # --------step1:加载智能体，在训练集上挑选句子-----------

    # 加载预训练模型
    net = torch.load('./model/pcnn_hme_pretrain.pkl')
    agent = torch.load('./model/instance_detector_pretrain.pkl')

    # 加载句子向量
    sent_embeddings = np.load('./model/pretrain_sentence_embedding.npy', allow_pickle=True)
    sent_tag = np.load('./model/sentence_predict_result.npy', allow_pickle=True)
    t_h = np.load('./model/t_h.npy', allow_pickle=True)

    # 现根据已预训练的智能体来对每个包选择句子
    print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ', epoch = ', epoch, '] step1: select credible instances (truth) from training data by pretrained instance detector\n')
    env = Environment(args.sent_dim)
    credible_train_word = []
    credible_train_pos1 = []
    credible_train_pos2 = []
    credible_train_y = []
    credible_train_entity_pair = []
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
            a.append(train_word[i][j])
            b.append(train_pos1[i][j])
            c.append(train_pos2[i][j])
        credible_train_word.append(a)
        credible_train_pos1.append(b)
        credible_train_pos2.append(c)
        credible_train_y.append(train_y[i])
        credible_train_entity_pair.append(train_entity_pair[i])
    print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ', epoch = ', epoch, '] step1: success for select credible instances\n')
    print('[noisy bag num : ', lost, ', bag num : ', len(sent_embeddings), ']')
    cre_train_word = np.array(credible_train_word)
    cre_train_pos1 = np.array(credible_train_pos1)
    cre_train_pos2 = np.array(credible_train_pos2)
    cre_train_y = np.array(credible_train_y)
    cre_train_entity_pair = np.array(credible_train_entity_pair)

    masks = process_mask(args, credible_train_word, padding_token, credible_train_entity_pair)
    masks2 = process_mask(args, test_word, padding_token, test_entity_pair)

    # --------step2:首先根据挑选的credible句子，训练PCNN+HME--------
    print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ', epoch = ', epoch, '] step2: training PCNN+HME on selected instances\n')
    for ep in range(1, 2):
        print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] pretraining PCNN+HME (epoch = ', ep, ')\n') 
        batches = batch_loader(cre_train_word, cre_train_pos1, cre_train_pos2, cre_train_y, np.array(masks), cre_train_entity_pair, shuffle=True)
        for e, batch_word_, batch_pos1_, batch_pos2_, batch_y_, batch_masks_, batch_entity_pair_ in batches:
            # print('batch_y_=', batch_y_)
            cotrain_step += 1
            net.optim.zero_grad()
            x, hie_y, scope, sent_label, head, tail, y_true = process_tensor(args, batch_word_, batch_pos1_, batch_pos2_, batch_masks_, batch_entity_pair_, batch_y_, hie_rel)
            y_pcnn_pred, y_hme_rel, _, loss = net.forward(x, hie_y, scope, sent_label, head, tail, train=True, hme_pro=False)
            
            loss.backward(retain_graph=True)
            net.optim.step()
            if (e + 1) % args.display_every == 0:
                log.print_pretrain_pcnnhme(ep, e+1, loss)

            if (cotrain_step + 1) % args.evaluate_every == 0:
                # 在测试集上进行测试，并获得当前最好的模型
                print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '] evaluting (epoch = ', ep, ')\n') 
                batches = batch_loader(test_word, test_pos1, test_pos2, test_y, np.array(masks2), test_entity_pair, shuffle=False)
                acc_h1, acc_h2, acc_h3, sum_ = 0, 0, 0, 0
                # sent_embeddings = []
                with torch.no_grad():
                    for e_, test_batch_word_, test_batch_pos1_, test_batch_pos2_, test_batch_y_, test_batch_masks_, test_batch_entity_pair_ in batches:
                        x, hie_y, scope, sent_label, head, tail, y_true = process_tensor(args, test_batch_word_, test_batch_pos1_, test_batch_pos2_, test_batch_masks_, test_batch_entity_pair_, test_batch_y_, hie_rel)
                        _, y_hme_rel, sent_embedding, _ = net.forward(x, hie_y, scope, sent_label, head, tail, train=False, hme_pro=False)
                        # hie_y表示每个包对应的三层标签，y_hme_rel则表示每个包预测的三层标签，注意，在测试阶段不会使用hie_y进行辅助增强，且不使用PCNN的预测结果
                        acc_sum_h1, acc_sum_h2, acc_sum_h3, test_sum = check_res(hie_y, y_hme_rel)
                        acc_h1 += acc_sum_h1
                        acc_h2 += acc_sum_h2
                        acc_h3 += acc_sum_h3
                        sum_ += test_sum
                        # sent_embeddings.append(sent_embedding)
                # sent_embeddings = torch.stack(sent_embeddings)
                ac_h1 = round(acc_h1*100.0/sum_, 2) # 相当于原始关系的精度（不计算NA的句子）
                ac_h2 = round(acc_h2*100.0/sum_, 2) # 只有两层关系的精度（不计算NA的句子）
                ac_h3 = round(acc_h3*100.0/sum_, 2) # 只有一层关系的精度（不计算NA的句子）
                if ac_h1 >= max_acc:
                    max_acc = ac_h1
                    torch.save(net, './model/pcnn_hme_pretrain.pkl') # 保存当前的模型
                log.print_evalue_pcnnhme(epoch, ep, ac_h1, ac_h2, ac_h3, acc_h1, acc_h2, acc_h3, sum_, max_acc)
                # print('evaluate acc: epoch:', epoch,' | acc (layer1) = ', ac_h1 ,' | acc (layer2) = ', ac_h2 ,' | acc (layer3) = ', ac_h3)
                # print('evaluate num: epoch:', epoch,' | accnum (layer1) = ', acc_h1 ,' | accnum (layer2) = ', acc_h2 ,' | accnum (layer3) = ', acc_h3, ' | testsum = ', sum_, '\n')

    # -----------step3:将预训练的模型重新加载回，并在训练集上对每个句子进行分类，获得每个句子的分类结果，并获得对应的句子表征-------------
    
    print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ', epoch = ', epoch, '] step3: obtain sentence-level embeddings and results based on the trained PCNN+HME\n')
    # net.load_state_dict(torch.load('./model/pcnn_hme_pretrain.pkl'))
    net = torch.load('./model/pcnn_hme_pretrain.pkl')
    sent_tag = [] # 保存每个包内每个句子被分类的结果，1表示当前句子分类正确，0表示当前句子分类错误
    sent_embeddings = [] # 保存每个包对应每个句子的表征向量
    t_h = []
    # 注意此处使用原始的数据集，因为需要知道所有句子的向量
    masks = process_mask(args, train_word, padding_token, train_entity_pair)
    batches = batch_loader(train_word, train_pos1, train_pos2, train_y, np.array(masks), train_entity_pair, shuffle=False)
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
    np.save('./model/pretrain_sentence_embedding.npy', sent_embeddings)
    np.save('./model/sentence_predict_result.npy', sent_tag)
    np.save('./model/t_h.npy', t_h)
    sent_embeddings = np.load('./model/pretrain_sentence_embedding.npy', allow_pickle=True)
    sent_tag = np.load('./model/sentence_predict_result.npy', allow_pickle=True)
    t_h = np.load('./model/t_h.npy', allow_pickle=True)


    # -----------step4:训练强化学习Agent-------------
    print('[', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ', epoch = ', epoch, '] step4: training Instance Detector')

    env = Environment(args.sent_dim)
    agent_epoch = 2 # 智能体训练次数
    for ep in range(agent_epoch):
        # 遍历每一个包（一个包内所有句子相当于一个episode）
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
                    action = agent.select_action(prob, is_epsilon=True) # 根据概率分布，带有贪心的进行执行动作
                    if j == len(sent_embeddings[i]) - 1: # 当前为幕的最后一个序列，需要单独计算延时奖励
                        reward = env.reward(tag)
                    else: # 其他时刻即时奖励为0 
                        reward = 0
                    agent.store_episode(current_state, action, reward) # 保存采样序列
                    current_state = env.step(action) # 根据当前的动作，环境转移到下一个状态，
            # step2:对采样的序列进行训练
            agent.train()
            if (i + 1) % (args.display_every * 100) == 0:
                log.print_pretrain_agent(ep, i+1, agent.loss)

    torch.save(agent, './model/instance_detector_pretrain.pkl') # 保存当前的模型