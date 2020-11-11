import numpy as np
import sys
import os
import datetime as dt
import sklearn.metrics
from configure import args
class Logger:
    def __init__(self, out_dir, timestamp):
        # 初始化日志加载信息
        self.log_dir = os.path.abspath(os.path.join(out_dir, 'logs'))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_path = os.path.abspath(os.path.join(self.log_dir, timestamp + '_logs.txt'))
        self.log_file = open(self.log_path, 'w')
        self.timestamp = timestamp



    def print_pretrain_pcnnhme(self, epoch, step, loss):
        # 打印训练过程记录
        if step == 1:
            self.log_file.write('\n===================== pretrain PCNN+HME ====================\n\n')
        time_str = dt.datetime.now().isoformat()
        train_log = '[{}]: epoch={} | step={} | loss={}\n'.format(time_str, epoch, step, loss)
        self.log_file.write(train_log)
        print(train_log)

    def print_evalue_pcnnhme(self, epoch, step, ac_h1, ac_h2, ac_h3, acc_h1, acc_h2, acc_h3, sum_, maxacc):
        # 打印训练过程记录
        if step == 1:
            self.log_file.write('\n===================== evaluate PCNN+HME ====================\n\n')
        time_str = dt.datetime.now().isoformat()
        log1 = '[{}]:evaluate acc: epoch={} | acc (layer1)={} % | acc (layer2)={} % | acc (layer3)={} % | maxacc (layer1)={} % | \n'.format(time_str, epoch, ac_h1, ac_h2, ac_h3, maxacc)
        log2 = '[{}]:evaluate num: epoch={} | accnum (layer1)={} | accnum (layer2)={} | accnum (layer3)={} | testsum={}\n'.format(time_str, epoch, acc_h1, acc_h2, acc_h3, sum_)
        self.log_file.write(log1)
        self.log_file.write(log2)
        print(log1)
        print(log2)

    def print_pretrain_agent(self, epoch, step, loss):
        # 打印训练过程记录
        if step == 1:
            self.log_file.write('\n===================== pretrain Instance Detector ====================\n\n')
        time_str = dt.datetime.now().isoformat()
        train_log = '[{}]: epoch={} | step={} | loss={}\n'.format(time_str, epoch, step, loss)
        self.log_file.write(train_log)
        print(train_log)

    def print_longtail(self, hits_10_acc, hits_15_acc, hits_20_acc):
        # 打印训练过程记录
        self.log_file.write('\n===================== evaluate long-tail relation  ====================\n\n')
        time_str = dt.datetime.now().isoformat()
        test_log = '[{}]: hits_10_acc={} | hits_15_acc={} | hits_20_acc={}\n'.format(time_str, hits_10_acc, hits_15_acc, hits_20_acc)
        self.log_file.write(test_log)
        print(test_log)

    def print_eval(self, step, test_result_l0, test_result_l1, test_result_l2, test_set_num, ac_h1, ac_h2, ac_h3):
        # 记录评估效果，并返回当前评估的auc值
        sorted_test_result_l0 = sorted(test_result_l0, key=lambda x: x['precision'])
        sorted_test_result_l1 = sorted(test_result_l1, key=lambda x: x['precision'])
        sorted_test_result_l2 = sorted(test_result_l2, key=lambda x: x['precision'])
        prec_l0, prec_l1, prec_l2 = [], [], []
        recall_l0, recall_l1, recall_l2 = [], [], []
        correct_l0 = 0
        correct_l1 = 0
        correct_l2 = 0
        # 倒序遍历(所有包数*关系类数)
        # PR计算方法：
        # 首先获得每个包的关系向量（属于该类元素为1，其余为0，形成一个矩阵），其次由模型得到每个包每个类的概率分布（也是一个矩阵），
        # 然后分别对这两个矩阵按行展开后转置形成两个列向量，然后按照从大到小排序，从第一个元素到最后一个元素开始，对应值每个作为一次阈值，即可计算P/R值
        for i, item in enumerate(sorted_test_result_l0[::-1]):
            correct_l0 += item['label']
            # 所有预测为正例的样本中预测正确的比例
            prec_l0.append(float(correct_l0) / (i + 1))
            # relfact_tot除NA关系外所有样本中关系的个数（单标签则恰好是样本数）
            # 所有实际为正例的样本中被预测为正例的比例
            recall_l0.append(float(correct_l0) / test_set_num)
        auc_l0 = sklearn.metrics.auc(x=recall_l0, y=prec_l0)

        for i, item in enumerate(sorted_test_result_l1[::-1]):
            correct_l1 += item['label']
            # 所有预测为正例的样本中预测正确的比例
            prec_l1.append(float(correct_l1) / (i + 1))
            # relfact_tot除NA关系外所有样本中关系的个数（单标签则恰好是样本数）
            # 所有实际为正例的样本中被预测为正例的比例
            recall_l1.append(float(correct_l1) / test_set_num)
        auc_l1 = sklearn.metrics.auc(x=recall_l1, y=prec_l1)

        for i, item in enumerate(sorted_test_result_l2[::-1]):
            correct_l2 += item['label']
            # 所有预测为正例的样本中预测正确的比例
            prec_l2.append(float(correct_l2) / (i + 1))
            # relfact_tot除NA关系外所有样本中关系的个数（单标签则恰好是样本数）
            # 所有实际为正例的样本中被预测为正例的比例
            recall_l2.append(float(correct_l2) / test_set_num)
        auc_l2 = sklearn.metrics.auc(x=recall_l2, y=prec_l2)

        self.log_file.write('\n====================== eval ====================\n\n')
        time_str = dt.datetime.now().isoformat()
        eval_log1 = '[{}]: step={} | acc (layer1)={} % | acc (layer2)={} % | acc (layer3)={} %\n'\
            .format(time_str, step, ac_h1, ac_h2, ac_h3)
        eval_log2 = '[{}]: step={} | auc (layer1)={} % | auc (layer2)={} % | auc (layer3)={} %\n'\
            .format(time_str, step, auc_l0, auc_l1, auc_l2)
        eval_log3 = '[{}]: step={} | P@100 (layer1)={} | P@200 (layer1)={} | P@300 (layer1)={}\n'\
            .format(time_str, step, prec_l0[99], prec_l0[199], prec_l0[299])
        eval_log4 = '[{}]: step={} | P@100 (layer2)={} | P@200 (layer2)={} | P@300 (layer2)={}\n'\
            .format(time_str, step, prec_l1[99], prec_l1[199], prec_l1[299])
        eval_log5 = '[{}]: step={} | P@100 (layer3)={} | P@200 (layer3)={} | P@300 (layer3)={}\n'\
            .format(time_str, step, prec_l2[99], prec_l2[199], prec_l2[299])
        self.log_file.write(eval_log1)
        self.log_file.write(eval_log2)
        self.log_file.write(eval_log3)
        self.log_file.write(eval_log4)
        self.log_file.write(eval_log5)
        print(eval_log1, eval_log2, eval_log3, eval_log4, eval_log5)
        np.save(args.model_dict + 'prec_layer_0.npy', prec_l0)
        np.save(args.model_dict + 'recall_layer_0.npy', recall_l0)
        np.save(args.model_dict + 'prec_layer_1.npy', prec_l1)
        np.save(args.model_dict + 'recall_layer_1.npy', recall_l1)
        np.save(args.model_dict + 'prec_layer_2.npy', prec_l2)
        np.save(args.model_dict + 'recall_layer_2.npy', recall_l2)
        
