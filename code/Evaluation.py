# coding: utf-8
# author: lu yf
# create date: 2018/11/15
from __future__ import division

import torch
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from tqdm import tqdm


class Evaluation:
    def __init__(self,emb_data, from_file):
        self.emb_data = emb_data

        self.id2emb = {}
        if from_file:
            self.format_training_data_from_file()
        else:
            self.id2emb = self.emb_data
        self.test_nodes_for_tlp = {}  # node: it's neighbors in 2017
        self.test_edges_for_tlp = {}
        self.node2his = {}
        self.candidate_nodes_for_recommendation = []

    def format_training_data_from_file(self):
        with open(self.emb_data, 'r') as reader:
            reader.readline()
            node_id = 0
            for line in reader:
                embeds = np.fromstring(line.strip(), dtype=float, sep=' ')
                self.id2emb[node_id] = embeds
                node_id += 1

    def lr_classification(self,label_data,train_ratio):
        i2l = dict()
        cl_x = []
        cl_y = []
        with open(label_data, 'r') as reader:
            for line in reader:
                parts = line.strip().split()
                n_id, l_id = int(parts[0]), int(parts[1])
                i2l[n_id] = l_id

        i2l_list = sorted(i2l.items(), key=lambda x:x[0])
        for (id, label) in i2l_list:
            cl_y.append(label)
            cl_x.append(self.id2emb[id])

        cl_x = np.stack(cl_x)
        x_train, x_valid, y_train, y_valid = train_test_split(cl_x, cl_y, test_size=1-train_ratio, random_state=9)
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_valid_pred = lr.predict(x_valid)
        micro_f1 = f1_score(y_valid, y_valid_pred, average='micro')
        macro_f1 = f1_score(y_valid, y_valid_pred, average='macro')
        print ('Macro_F1_score:{}'.format(macro_f1))
        print ('Micro_F1_score:{}'.format(micro_f1))

    def network_reconstruction(self,reconstruction_file,train_ratio):
        nr_x = []
        nr_y = []
        with open(reconstruction_file,'r') as re_file:
            for line in re_file:
                tokens = line.strip().split(' ')
                i_emb = self.id2emb[int(tokens[0])]
                j_emb = self.id2emb[int(tokens[1])]
                nr_x.append(abs(i_emb-j_emb))
                nr_y.append(int(tokens[2]))

        x_train, x_valid, y_train, y_valid = train_test_split(nr_x, nr_y, test_size=1 - train_ratio, random_state=9)
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_valid_pred_prob = lr.predict_proba(x_valid)[:,1]
        y_valid_pred_01 = lr.predict(x_valid)
        acc = accuracy_score(y_valid, y_valid_pred_01)
        auc = roc_auc_score(y_valid, y_valid_pred_prob)
        f1 = f1_score(y_valid, y_valid_pred_01)
        print ('acc:{}'.format(acc))
        print ('auc:{}'.format(auc))
        print ('f1:{}'.format(f1))


if __name__ == '__main__':
    # dblp
    eva = Evaluation(emb_data='../res/dblp/tne_epoch50_lr0.01_his5_neg5.emb', from_file=True)
    eva.lr_classification(train_ratio=0.8, label_data='../data/dblp/node2label.txt')

    # #tmall
    eva = Evaluation(emb_data='../res/tmall/tne_epoch200_lr0.01_his2_neg5.emb', from_file=True)
    eva.lr_classification(train_ratio=0.8, label_data='../data/tmall/node2label.txt')