# coding: utf-8
# author: lu yf
# create date: 2018/11/12

from __future__ import division

import time

import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import sys
from DataHelper import DataHelper
from Evaluation import Evaluation
import os


FType = torch.FloatTensor
LType = torch.LongTensor

DID = 0


class MMDNE:
    def __init__(self, file_path, cl_label_data,nr_data,save_path, emb_size=128, neg_size=10, hist_len=2, directed=False,
                 learning_rate=0.01, batch_size=1000, save_step=10, epoch_num=1, optim='SGD',
                 tlp_flag=False, trend_prediction=False, epsilon=1.0):
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num
        self.cl_label_data = cl_label_data
        self.nr_data = nr_data
        self.save_path = save_path

        self.optim = optim
        self.tlp_flag = tlp_flag
        self.trend_prediction = trend_prediction

        self.epsilon = epsilon

        print ('dataset helper...')
        self.data = DataHelper(file_path, neg_size, hist_len, directed, tlp_flag=self.tlp_flag, trend_pred_flag=self.trend_prediction)
        self.node_dim = self.data.get_node_dim()
        print ('max time: {}'.format(self.data.get_max_d_time()))

        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                    FType).cuda(), requires_grad=True)

                self.delta_s = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.delta_t = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)

                self.zeta = Variable((torch.ones(1)).type(FType).cuda(), requires_grad=True)
                self.gamma = Variable((torch.ones(1)).type(FType).cuda(), requires_grad=True)
                self.theta = Variable((torch.ones(1)).type(FType).cuda(), requires_grad=True)

                self.gat_hidden_size = 128

                self.global_att_linear_layer = torch.nn.Linear(self.gat_hidden_size, 1).cuda()

                self.W = torch.nn.Parameter(torch.zeros(size=(self.emb_size, self.gat_hidden_size))).cuda()
                torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
                self.a = torch.nn.Parameter(torch.zeros(size=(2 * self.gat_hidden_size, 1))).cuda()
                torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

                self.leakyrelu = torch.nn.LeakyReLU(0.2)  # alpha =0.2 for leakyrelu

        else:
            print 'no gpu'
            self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                FType), requires_grad=True)

            self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)

            self.att_param = Variable(torch.diag(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(emb_size), 1. / np.sqrt(emb_size), (emb_size,))).type(
                FType)), requires_grad=True)

        if self.optim == 'SGD':
            self.opt = SGD(lr=learning_rate, params=[self.node_emb, self.delta_s, self.delta_t,
                                                     self.zeta,self.gamma,self.theta])
        elif self.optim == 'Adam':
            self.opt = Adam(lr=learning_rate, params=[self.node_emb, self.delta_s, self.delta_t,
                                                      self.zeta,self.gamma,self.theta])

        self.loss = torch.FloatTensor()
        self.micro_loss = torch.FloatTensor()
        self.macro_loss = torch.FloatTensor()
        self.global_attention = Variable((torch.zeros(1)).type(FType).cuda())

    def local_forward(self, s_nodes, t_nodes, e_times,
                      s_h_nodes, s_h_times, s_h_time_mask,
                      t_h_nodes, t_h_times, t_h_time_mask,
                      s_neg_node,t_neg_node):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)  # (bach, emb_dim)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        s_h_node_emb = self.node_emb.index_select(0, Variable(s_h_nodes.view(-1))).view(batch, self.hist_len,-1)
        t_h_node_emb = self.node_emb.index_select(0, Variable(t_h_nodes.view(-1))).view(batch, self.hist_len, -1)

        delta_s = self.delta_s.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)  # (b,1)
        delta_t = self.delta_s.index_select(0, Variable(t_nodes.view(-1))).unsqueeze(1)  # TODO: delta_t ???
        d_time_s = torch.abs(e_times.unsqueeze(1) - s_h_times)  # (batch, hist_len)
        d_time_t = torch.abs(e_times.unsqueeze(1) - t_h_times)  # (batch, hist_len)

        # GAT attention_rewrite
        for i in xrange(self.hist_len):
            s_h_node_emb_i = torch.transpose(s_h_node_emb[:, i:(i + 1), :], dim0=1, dim1=2).squeeze()  # (b, dim)
            s_node_emb_i = s_node_emb  # (b, dim)
            d_time_s_i = Variable(d_time_s)[:,i:(i+1)]  # (b,1)
            if i == 0:
                a_input = torch.cat([torch.mm(s_node_emb_i, self.W),torch.mm(s_h_node_emb_i, self.W)],dim=1)  # (b, 2*dim)
                sim_s_s_his = self.leakyrelu(torch.exp(-delta_s * d_time_s_i) * torch.mm(a_input, self.a))  # (b.dim)
            else:
                a_input = torch.cat([torch.mm(s_node_emb_i, self.W),torch.mm(s_h_node_emb_i, self.W)],dim=1)
                sim_s_s_his = torch.cat([sim_s_s_his,
                                         self.leakyrelu(torch.exp(-delta_s * d_time_s_i) * torch.mm(a_input, self.a))], dim=1)

        for i in xrange(self.hist_len):
            t_h_node_emb_i = torch.transpose(t_h_node_emb[:, i:(i + 1), :], dim0=1, dim1=2).squeeze()
            t_node_emb_i = t_node_emb
            d_time_t_i = Variable(d_time_t)[:, i:(i + 1)]  # (b,1)
            if i == 0:
                a_input = torch.cat([torch.mm(t_node_emb_i, self.W), torch.mm(t_h_node_emb_i, self.W)],
                                    dim=1)  # (b, 2*dim)
                sim_t_t_his = self.leakyrelu(torch.exp(-delta_s * d_time_t_i) * torch.mm(a_input, self.a))  # (b, 1)
            else:
                a_input = torch.cat([torch.mm(t_node_emb_i, self.W), torch.mm(t_h_node_emb_i, self.W)], dim=1)
                sim_t_t_his = torch.cat([sim_t_t_his,
                                         self.leakyrelu(torch.exp(-delta_s * d_time_t_i) * torch.mm(a_input, self.a))],
                                        dim=1)

        att_s_his_s = softmax(sim_s_s_his, dim=1)  # (batch, h)
        att_t_his_t = softmax(sim_t_t_his, dim=1)  # (batch, h)

        s_his_hat_emb_inter = ((att_s_his_s * Variable(s_h_time_mask)).unsqueeze(2) *
                               torch.mm(s_h_node_emb.view(s_h_node_emb.size()[0] * self.hist_len, -1), self.W).
                               view(s_h_node_emb.size()[0],self.hist_len,-1)).sum(dim=1)  # (b,dim)
        t_his_hat_emb_inter = ((att_t_his_t * Variable(t_h_time_mask)).unsqueeze(2) *
                               torch.mm(t_h_node_emb.view(t_h_node_emb.size()[0] * self.hist_len, -1), self.W).
                               view(t_h_node_emb.size()[0],self.hist_len,-1)).sum(dim=1)

        # temporal-self-attention
        global_att = softmax(torch.tanh(self.global_att_linear_layer(torch.transpose(
            torch.cat([(s_his_hat_emb_inter * torch.exp(-delta_s * Variable(d_time_s.mean(dim=1)).unsqueeze(1))).unsqueeze(2),
                       (t_his_hat_emb_inter * torch.exp(-delta_t * Variable(d_time_t.mean(dim=1)).unsqueeze(1))).unsqueeze(2)],
                      dim=2),dim0=1,dim1=2))),dim=1).squeeze(2)  # (dim, 2)
        global_att_s = global_att[:, 0]
        global_att_t = global_att[:, 1]
        self.global_attention = global_att

        # global_att_s = 0.5
        # global_att_t = 0.5
        p_mu = ((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg()  # (batch, 1)
        p_alpha_s = ((s_h_node_emb - t_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()   # (batch, h_len)
        p_alpha_t = ((t_h_node_emb - s_node_emb.unsqueeze(1)) ** 2).sum(dim=2).neg()

        p_lambda = p_mu \
                   + global_att_s * (att_s_his_s * p_alpha_s * torch.exp(delta_s * Variable(d_time_s)) * Variable(s_h_time_mask)).sum(
            dim=1) \
                   + global_att_t * (att_t_his_t * p_alpha_t * torch.exp(delta_t * Variable(d_time_t)) * Variable(t_h_time_mask)).sum(
            dim=1)

        s_n_node_emb = self.node_emb.index_select(0, Variable(s_neg_node.view(-1))).view(batch, self.neg_size, -1)
        t_n_node_emb = self.node_emb.index_select(0, Variable(t_neg_node.view(-1))).view(batch, self.neg_size, -1)

        n_mu_s = ((s_node_emb.unsqueeze(1) - t_n_node_emb) ** 2).sum(dim=2).neg()  # (batch, neg_len)
        n_mu_t = ((t_node_emb.unsqueeze(1) - s_n_node_emb) ** 2).sum(dim=2).neg()
        n_alpha_s = ((s_h_node_emb.unsqueeze(2) - t_n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()
        n_alpha_t = ((t_h_node_emb.unsqueeze(2) - s_n_node_emb.unsqueeze(1)) ** 2).sum(dim=3).neg()

        n_lambda_s = n_mu_s \
                     + global_att_s.unsqueeze(1) * (att_s_his_s.unsqueeze(2) * n_alpha_s
                                                    * (torch.exp(delta_s * Variable(d_time_s)).unsqueeze(2))
                                                    * (Variable(s_h_time_mask).unsqueeze(2))).sum(dim=1)   # TODO: global_att_s.unsqueeze(1)

        n_lambda_t = n_mu_t \
                     + global_att_t.unsqueeze(1) * (att_t_his_t.unsqueeze(2) * n_alpha_t
                                                    * (torch.exp(delta_t * Variable(d_time_t)).unsqueeze(2))
                                                    * (Variable(t_h_time_mask).unsqueeze(2))).sum(dim=1)

        return p_lambda, n_lambda_s, n_lambda_t  # max p_lambda, min n_lambda

    def global_forward(self, s_nodes, t_nodes, e_times, delta_n_true, node_sum, edge_last_time_sum):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)  # (bach, emb_dim)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)

        beta = torch.sigmoid(((s_node_emb - t_node_emb) ** 2).sum(dim=1).neg())  # (batch,1)

        delta_e_pred = beta / torch.pow(Variable(e_times)+1e-6,self.theta) * Variable(node_sum) * \
                       (self.zeta * torch.pow(Variable(node_sum-1),self.gamma))

        return delta_e_pred

    def local_loss(self, s_nodes, t_nodes, e_times,
                   s_h_nodes, s_h_times, s_h_time_mask,
                   t_h_nodes, t_h_times, t_h_time_mask,
                   neg_s_node,neg_t_node):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                p_lambdas, n_lambdas_s, n_lambdas_t = self.local_forward(s_nodes, t_nodes, e_times,
                                                                                     s_h_nodes, s_h_times,
                                                                                     s_h_time_mask,
                                                                                     t_h_nodes, t_h_times,
                                                                                     t_h_time_mask,
                                                                                     neg_s_node, neg_t_node)

                loss = - torch.log(p_lambdas.sigmoid() + 1e-6) \
                       - torch.log(n_lambdas_s.neg().sigmoid() + 1e-6).sum(dim=1) \
                       - torch.log(n_lambdas_t.neg().sigmoid() + 1e-6).sum(dim=1)

        else:
            p_lambdas, n_lambdas_s, n_lambdas_t = self.forward(s_nodes, t_nodes, e_times,
                                                               s_h_nodes, s_h_times, s_h_time_mask,
                                                               t_h_nodes, t_h_times, t_h_time_mask,
                                                               s_neg_node,t_neg_node)
            loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) \
                   - torch.log(torch.sigmoid(torch.neg(n_lambdas_s)) + 1e-6).sum(dim=1) \
                   - torch.log(torch.sigmoid(torch.neg(n_lambdas_t)) + 1e-6).sum(dim=1)
        return loss

    def global_loss(self,s_nodes, t_nodes, e_times, delta_e_true, delta_n_true, node_sum, edge_last_time_sum):
        with torch.cuda.device(DID):
            delta_e_pred = self.global_forward(s_nodes, t_nodes, e_times, delta_n_true, node_sum, edge_last_time_sum)
            criterion = torch.nn.MSELoss()
            loss = criterion(torch.log(delta_e_pred + 1e-6), torch.log(Variable(delta_e_true) + 1e-6))
        return loss

    def update(self, s_nodes, t_nodes, e_times,
               s_h_nodes, s_h_times, s_h_time_mask,
               t_h_nodes, t_h_times, t_h_time_mask,
               neg_s_node,neg_t_node,
               delta_e_true, delta_n_true, node_sum, edge_last_time_sum):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                local_loss = self.local_loss(s_nodes, t_nodes, e_times,
                                             s_h_nodes, s_h_times, s_h_time_mask,
                                             t_h_nodes, t_h_times, t_h_time_mask,
                                             neg_s_node, neg_t_node)

                global_loss = self.global_loss(s_nodes, t_nodes, e_times,
                                               delta_e_true, delta_n_true, node_sum, edge_last_time_sum)
                loss = (1-self.epsilon)*local_loss.sum() + self.epsilon * global_loss.sum()

                self.loss += loss.data
                self.micro_loss += local_loss.sum().data
                self.macro_loss += global_loss.sum().data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            local_loss = self.local_loss(s_nodes, t_nodes, e_times,
                                         s_h_nodes, s_h_times, s_h_time_mask,
                                         t_h_nodes, t_h_times, t_h_time_mask)

            global_loss = self.global_loss(s_nodes, t_nodes, e_times,
                                           delta_e_true, delta_n_true, node_sum, edge_last_time_sum)
            loss = local_loss.sum() + global_loss.sum()

            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        print ('training...')
        for epoch in xrange(self.epochs):
            self.loss = 0.0
            self.micro_loss = 0.0
            self.macro_loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch,
                                shuffle=True, num_workers=10)
            if epoch % self.save_step == 0 and epoch != 0:
                emb_save_name = os.path.join(self.save_path, ('tne_epoch%d_lr%.2f_his%d_neg%d_eps%.1f.emb'
                                                                  % (epoch, self.lr, self.hist_len, self.neg_size,self.epsilon)))
                self.save_node_embeddings(emb_save_name)

            if epoch % 10 == 0 and epoch != 0:
                print ('evaluation...')
                if torch.cuda.is_available():
                    embeddings = self.node_emb.cpu().data.numpy()
                else:
                    embeddings = self.node_emb.data.numpy()
                eva = Evaluation(emb_data=embeddings,from_file=False)
                eva.lr_classification(train_ratio=0.8,label_data=self.cl_label_data)

            for i_batch, sample_batched in enumerate(loader):
                if i_batch % 1000 == 0 and i_batch != 0:
                    sys.stdout.write('\r\n' + str(i_batch * self.batch)
                                     + '\tloss: ' + str(self.loss.cpu().numpy() / (self.batch * i_batch))
                                     # + '\tmicro_loss: ' + str(self.micro_loss.cpu().numpy() / (self.batch * i_batch))
                                     # + '\tmacro_loss: ' + str(self.macro_loss.cpu().numpy() / (self.batch * i_batch))
                                     # + '\r\ndelta_s:' + str(self.delta_s.mean().cpu().data.numpy())
                                     # + '\tdelta_t:' + str(self.delta_t.mean().cpu().data.numpy())
                                     # + '\tglobal_attention:' + str(self.global_attention.mean(dim=0).cpu().data.numpy())
                                     # + '\r\nzeta:' + str(self.zeta.mean().cpu().data.numpy())
                                     # + '\tgamma:' + str(self.gamma.mean().cpu().data.numpy())
                                     # + '\ttheta:' + str(self.theta.mean().cpu().data.numpy())
                    )
                    sys.stdout.flush()

                if torch.cuda.is_available():
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['event_time'].type(FType).cuda(),
                                    sample_batched['s_history_nodes'].type(LType).cuda(),
                                    sample_batched['s_history_times'].type(FType).cuda(),
                                    sample_batched['s_history_masks'].type(FType).cuda(),
                                    sample_batched['t_history_nodes'].type(LType).cuda(),
                                    sample_batched['t_history_times'].type(FType).cuda(),
                                    sample_batched['t_history_masks'].type(FType).cuda(),
                                    sample_batched['neg_s_nodes'].type(LType).cuda(),
                                    sample_batched['neg_t_nodes'].type(LType).cuda(),
                                    sample_batched['delta_e_true'].type(FType).cuda(),
                                    sample_batched['delta_n_true'].type(FType).cuda(),
                                    sample_batched['node_sum'].type(FType).cuda(),
                                    sample_batched['edge_last_time_sum'].type(FType).cuda())
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['event_time'].type(FType),
                                sample_batched['s_neg_nodes'].type(LType),
                                sample_batched['s_history_nodes'].type(LType),
                                sample_batched['s_history_times'].type(FType),
                                sample_batched['s_history_masks'].type(FType),
                                sample_batched['t_neg_nodes'].type(LType),
                                sample_batched['t_history_nodes'].type(LType),
                                sample_batched['t_history_times'].type(FType),
                                sample_batched['t_history_masks'].type(FType))

            sys.stdout.write('\r\nepoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data)) + '\n')
            sys.stdout.flush()

        emb_save_name = os.path.join(self.save_path, ('tne_epoch%d_lr%.2f_his%d_neg%d_eps%.1f.emb'
                                                          % (self.epochs, self.lr, self.hist_len, self.neg_size,self.epsilon)))
        self.save_node_embeddings(emb_save_name)

        print ('evaluation...')
        eva = Evaluation(emb_data=emb_save_name, from_file=True)
        eva.lr_classification(train_ratio=0.8, label_data=self.cl_label_data)

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in xrange(self.node_dim):
            writer.write(' '.join(str(d) for d in embeddings[n_idx]) + '\n')
        writer.close()


if __name__ == '__main__':
    print time.asctime(time.localtime(time.time()))
    parameters_dict = {
        'file_path': '../data/dblp/dblp.txt',
        'cl_label_data': '../data/dblp/node2label.txt',
        'nr_data': '../data/dblp/dblp_network_reconstruction_0.01edge.txt',
        'save_path': '../res/dblp/',
        'epoch_num': 100,
        'batch_size': 100,
        'emb_size': 128,
        'learning_rate': 0.02,
        'neg_size': 5,
        'hist_len': 5,
        'directed': False,
        'save_step': 50,
        'optimization': 'SGD',
        'tlp_flag':False,
        'trend_prediction':False,
        'epsilon':0.4}
    print ('parameters: \r\n{}'.format(parameters_dict))

    mmdne = MMDNE(file_path=parameters_dict['file_path'],
                  cl_label_data=parameters_dict['cl_label_data'],
                  nr_data=parameters_dict['nr_data'],
                  save_path=parameters_dict['save_path'],
                  save_step=parameters_dict['save_step'],
                  directed=parameters_dict['directed'],
                  epoch_num=parameters_dict['epoch_num'],
                  hist_len=parameters_dict['hist_len'],
                  neg_size=parameters_dict['neg_size'],
                  learning_rate=parameters_dict['learning_rate'],
                  batch_size=parameters_dict['batch_size'],
                  optim=parameters_dict['optimization'],
                  tlp_flag=parameters_dict['tlp_flag'],
                  trend_prediction=parameters_dict['trend_prediction'],
                  epsilon=parameters_dict['epsilon'])

    mmdne.train()

    print ('parameters: \r\n{}'.format(parameters_dict))


