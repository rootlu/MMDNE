# coding: utf-8
# author: lu yf
# create date: 2018/11/12

from __future__ import division
from torch.utils.data import Dataset
import numpy as np
import sys
import random
import copy
np.random.seed(1)


class DataHelper(Dataset):
    def __init__(self, file_path, neg_size, hist_len, directed=False, transform=None, tlp_flag=False, trend_pred_flag=False):
        self.node2hist = dict()
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.directed = directed
        self.transform = transform

        self.max_d_time = -sys.maxint  # Time interval [0, T]

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)

        self.node_time_nodes = dict()
        self.node_set = set()
        self.degrees = dict()
        self.edge_list = []

        self.node_rate = {}
        self.edge_rate = {}
        self.node_sum = {}
        self.edge_sum = {}
        self.time_stamp = []
        self.time_edges_dict = {}
        self.time_nodes_dict = {}
        print ('loading data...')
        with open(file_path, 'r') as infile:
            for line in infile:
                parts = line.strip().split()
                s_node = int(parts[0])  # source node
                t_node = int(parts[1])  # target node
                d_time = float(parts[2])  # time slot, delta t

                self.node_set.update([s_node, t_node])  # node set

                if s_node not in self.degrees:
                    self.degrees[s_node] = 0
                if t_node not in self.degrees:
                    self.degrees[t_node] = 0

                # dblp temporal link prediction
                if tlp_flag:
                    if d_time >= 1.0:  # 2017 year
                        continue
                # # eucore temporal link prediction
                # if tlp_flag:
                #     if d_time >= 0.631382316314:  # 2017 year
                #         continue

                # # dblp Trend Prediction
                # if trend_pred_flag:
                #     if d_time > 0.5:
                #         continue
                # # tmall Trend Prediction
                # if trend_pred_flag:
                #     if d_time > 0.729317:
                #         continue
                # eucore Trend Prediction
                if trend_pred_flag:
                    if d_time > 0.333748443337:
                        continue

                self.edge_list.append((s_node,t_node,d_time))  # edge list
                if not directed:
                    self.edge_list.append((t_node,s_node,d_time))

                if s_node not in self.node2hist:
                    self.node2hist[s_node] = list()
                self.node2hist[s_node].append((t_node, d_time))
                if not directed:  # undirected
                    if t_node not in self.node2hist:
                        self.node2hist[t_node] = list()
                    self.node2hist[t_node].append((s_node, d_time))

                if s_node not in self.node_time_nodes:
                    self.node_time_nodes[s_node] = dict()
                if d_time not in self.node_time_nodes[s_node]:
                    self.node_time_nodes[s_node][d_time] = list()
                self.node_time_nodes[s_node][d_time].append(t_node)
                if not directed:  # undirected
                    if t_node not in self.node_time_nodes:
                        self.node_time_nodes[t_node] = dict()
                    if d_time not in self.node_time_nodes[t_node]:
                        self.node_time_nodes[t_node][d_time] = list()
                    self.node_time_nodes[t_node][d_time].append(s_node)

                if d_time > self.max_d_time:
                    self.max_d_time = d_time  # record the max time

                self.degrees[s_node] += 1
                self.degrees[t_node] += 1

                self.time_stamp.append(d_time)
                if not self.time_edges_dict.has_key(d_time):
                    self.time_edges_dict[d_time] = []
                self.time_edges_dict[d_time].append((s_node, t_node))
                if not self.time_nodes_dict.has_key(d_time):
                    self.time_nodes_dict[d_time] = []
                self.time_nodes_dict[d_time].append(s_node)
                self.time_nodes_dict[d_time].append(t_node)

        self.time_stamp = sorted(list(set(self.time_stamp)))  # !!! time from 0 to 1

        self.node_dim = len(self.node_set)  # number of nodes 28085

        self.data_size = 0  # number of edges, undirected x2 = 473788
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist, key=lambda x: x[1])  # from past(0) to now(1)
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.max_nei_len = max(map(lambda x: len(x), self.node2hist.values()))  # 955
        print ('#nodes: {}, #edges: {}, #time_stamp: {}'.
               format(self.node_dim,len(self.edge_list),len(self.time_stamp)))
        print ('avg. degree: {}'.format(sum(self.degrees.values())/len(self.degrees)))
        print ('max neighbors length: {}'.format(self.max_nei_len))
        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            for t_idx in xrange(len(self.node2hist[s_node])):
                self.idx2source_id[idx] = s_node
                self.idx2target_id[idx] = t_idx
                idx += 1

        print ('get edge rate...')
        self.get_edge_rate()

        print ('init. neg_table...')
        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()

    def get_edge_rate(self):
        for i in xrange(len(self.time_stamp)):
            current_nodes = []
            current_edges = []
            current_time_idx = i
            while current_time_idx >= 0:
                current_nodes += self.time_nodes_dict[self.time_stamp[current_time_idx]]
                current_edges += self.time_edges_dict[self.time_stamp[current_time_idx]]
                current_time_idx -= 1
            self.node_sum[self.time_stamp[i]] = len(set(current_nodes))
            self.edge_sum[self.time_stamp[i]] = len(current_edges)

        for i in xrange(len(self.time_stamp)):
            current_time_idx = i
            if current_time_idx == 0:  # time = 0, delta_node = node_sum[0]
                self.node_rate[self.time_stamp[current_time_idx]] = self.node_sum[self.time_stamp[current_time_idx]]
            else:
                self.node_rate[self.time_stamp[current_time_idx]] = \
                    self.node_sum[self.time_stamp[current_time_idx]] - self.node_sum[self.time_stamp[current_time_idx-1]]
            if current_time_idx == 0:
                self.edge_rate[self.time_stamp[current_time_idx]] = self.edge_sum[
                    self.time_stamp[current_time_idx]]
            else:
                self.edge_rate[self.time_stamp[current_time_idx]] = \
                    self.edge_sum[self.time_stamp[current_time_idx]] - self.edge_sum[
                        self.time_stamp[current_time_idx - 1]]

    def get_node_dim(self):
        return self.node_dim

    def get_max_d_time(self):
        return self.max_d_time

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        for k in xrange(self.node_dim):
            tot_sum += np.power(self.degrees[k], self.NEG_SAMPLING_POWER)
        for k in xrange(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def get_histories(self,node,remove_node,time):
        lack_hist_num = self.hist_len
        current_time_idx = self.time_stamp.index(time)
        hist_nodes = []
        hist_times = []
        while lack_hist_num > 0 and current_time_idx >= 0:
            current_nodes = copy.copy(self.node_time_nodes[node][self.time_stamp[current_time_idx]])  # !!! deep copy
            if current_time_idx == self.time_stamp.index(time):  # remove target node at current time
                current_nodes.remove(remove_node)
            if current_nodes is None:
                current_nodes = []
            if len(current_nodes) + len(hist_nodes) >= self.hist_len:
                hist_nodes += random.sample(current_nodes, lack_hist_num)
                hist_times += [self.time_stamp[current_time_idx]] * lack_hist_num
                break
            else:
                hist_nodes += current_nodes
                hist_times += [self.time_stamp[current_time_idx]] * len(current_nodes)
                lack_hist_num -= len(current_nodes)

                current_time_idx -= 1
                while not self.node_time_nodes[node].has_key(self.time_stamp[current_time_idx]):
                    current_time_idx -= 1

        np_his_nodes = np.zeros((self.hist_len,))
        np_his_nodes[:len(hist_nodes)] = hist_nodes
        np_his_times = np.zeros((self.hist_len,))
        np_his_times[:len(hist_times)] = hist_times
        np_his_masks = np.zeros((self.hist_len,))
        np_his_masks[:len(hist_nodes)] = 1.

        return np_his_nodes, np_his_times, np_his_masks

    def get_histories_for_gat(self,node,remove_node,time):
        current_time_idx = self.time_stamp.index(time)
        hist_nodes = []
        hist_times = []
        while current_time_idx >= 0:
            current_nodes = copy.copy(self.node_time_nodes[node][self.time_stamp[current_time_idx]])  # !!! deep copy
            if current_time_idx == self.time_stamp.index(time):  # remove target node at current time
                current_nodes.remove(remove_node)
            if current_nodes is None:
                current_nodes = []
            hist_nodes += current_nodes
            hist_times += [self.time_stamp[current_time_idx]] * len(current_nodes)
            current_time_idx -= 1
            while not self.node_time_nodes[node].has_key(self.time_stamp[current_time_idx]):
                current_time_idx -= 1

        np_his_nodes = np.zeros((self.max_nei_len,))
        np_his_nodes[:len(hist_nodes)] = hist_nodes
        np_his_times = np.zeros((self.max_nei_len,))
        np_his_times[:len(hist_times)] = hist_times
        np_his_masks = np.zeros((self.max_nei_len,))
        np_his_masks[:len(hist_nodes)] = 1.

        return np_his_nodes, np_his_times, np_his_masks

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # sampling via htne
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]
        t_node = self.node2hist[s_node][t_idx][0]
        e_time = self.node2hist[s_node][t_idx][1]
        if t_idx - self.hist_len < 0:
            s_his = self.node2hist[s_node][0:t_idx]
        else:
            s_his = self.node2hist[s_node][t_idx - self.hist_len:t_idx]

        # get the history neighbors for target node
        t_his_list = self.node2hist[t_node]
        s_idx = t_his_list.index((s_node, e_time))
        if s_idx - self.hist_len < 0:
            t_his = t_his_list[:s_idx]
        else:
            t_his = t_his_list[s_idx - self.hist_len:s_idx]

        s_his_nodes = np.zeros((self.hist_len,))
        s_his_nodes[:len(s_his)] = [h[0] for h in s_his]
        s_his_times = np.zeros((self.hist_len,))
        s_his_times[:len(s_his)] = [h[1] for h in s_his]
        s_his_masks = np.zeros((self.hist_len,))
        s_his_masks[:len(s_his)] = 1.

        t_his_nodes = np.zeros((self.hist_len,))
        t_his_nodes[:len(t_his)] = [h[0] for h in t_his]
        t_his_times = np.zeros((self.hist_len,))
        t_his_times[:len(t_his)] = [h[1] for h in t_his]
        t_his_masks = np.zeros((self.hist_len,))
        t_his_masks[:len(t_his)] = 1.

        # negative sampling
        neg_s_nodes = self.negative_sampling()
        neg_t_nodes = self.negative_sampling()

        time_idx = self.time_stamp.index(e_time)
        delta_e_true = self.edge_rate[e_time]
        delta_n_true = self.node_rate[e_time]
        node_sum = self.node_sum[e_time]
        if time_idx >= 1:
            edge_last_time_sum = self.edge_sum[self.time_stamp[time_idx-1]]
        else:
            edge_last_time_sum = self.edge_sum[self.time_stamp[time_idx]]

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'event_time': e_time,
            's_history_nodes': s_his_nodes,
            't_history_nodes': t_his_nodes,
            's_history_times': s_his_times,
            't_history_times': t_his_times,
            's_history_masks': s_his_masks,
            't_history_masks': t_his_masks,
            'neg_s_nodes': neg_s_nodes,
            'neg_t_nodes': neg_t_nodes,
            'delta_e_true': delta_e_true,
            'delta_n_true': delta_n_true,
            'node_sum': node_sum,
            'edge_last_time_sum': edge_last_time_sum
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
        sampled_nodes = self.neg_table[rand_idx]
        return sampled_nodes

