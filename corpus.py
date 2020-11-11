# 考虑到原始给定的词向量有些实体并不存在对应的词向量，因此需要单独重新训练
# 将训练集和测试集整合为一个大的语料，基于此进行GloVe词向量的训练。
# 本程序用于对训练集和测试集中所有句子进行汇总。生成corpus.txt后直接放入开源的GloVe项目中即可
# 数据集样例：
# m.01l443l    m.04t_bj    dave_holland    barry_altschul    NA    the occasion was suitably exceptional : a reunion of the 1970s-era sam rivers trio , with dave_holland on bass and barry_altschul on drums .    ###END###
import numpy as np
sent = []
with open('./origin_data/train.txt', 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for i in lines:        
        sent.append(i.replace(' ###END###', '').split('\t')[-1])
with open('./origin_data/test.txt', 'r', encoding='utf-8') as fr:
    lines = fr.readlines()
    for i in lines:        
        sent.append(i.replace(' ###END###', '').split('\t')[-1])
print(sent[13000:13010])
with open('./corpus/corpus.txt', 'w', encoding='utf-8') as fw:
    for i in sent:        
        fw.write(i)


