

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class G2A(nn.Module):
    def __init__(self,input_shape, args):
        super(G2A, self).__init__()
        self.args = args
        # Encoding
        self.n_agents = args.n_agents
        self.attention_dim = args.n_attention
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.encoding = nn.Linear(input_shape-self.n_agents, self.rnn_hidden_dim)
        self.h = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        # Hard
        self.hard_bi_GRU = nn.GRU(self.rnn_hidden_dim * 2, self.rnn_hidden_dim, bidirectional=True)
        self.hard_encoding = nn.Linear(self.rnn_hidden_dim * 2, 2)

        # Soft
        self.q = nn.Linear(self.rnn_hidden_dim, self.attention_dim, bias=False)
        self.k = nn.Linear(self.rnn_hidden_dim, self.attention_dim, bias=False)
        self.v = nn.Linear(self.rnn_hidden_dim, self.attention_dim)

        self.input_shape = input_shape


    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(self.n_agents, self.args.rnn_hidden_dim)


    def forward(self, obs, hidden_state):

        size = obs.shape[0] # batch_size * n_agents
        obs_encoding = f.relu(self.encoding(obs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)

        # GRU
        h_out = self.h(obs_encoding,h_in)

        # Hard Attention
        h = h_out.reshape(-1, self.n_agents,self.rnn_hidden_dim)
        input_hard = []

        for i in range(self.n_agents):
            h_i = h[:, i]  # (batch_size, rnn_hidden_dim)
            h_hard_i = []
            for j in range(self.n_agents):
                if j != i:
                    h_hard_i.append(torch.cat([h_i, h[:, j]], dim=-1))
            h_hard_i = torch.stack(h_hard_i, dim=0)
            input_hard.append(h_hard_i)

        input_hard = torch.stack(input_hard,dim=-2)
        input_hard = input_hard.view(self.n_agents - 1, -1, self.rnn_hidden_dim * 2)
        h_hard = torch.zeros((2 * 1, size, self.rnn_hidden_dim))
        h_hard, _ = self.hard_bi_GRU(input_hard,h_hard)
        h_hard = h_hard.permute(1, 0, 2)
        h_hard = h_hard.reshape(-1,self.rnn_hidden_dim * 2)
        hard_weights = self.hard_encoding(h_hard)
        hard_weights = f.gumbel_softmax(hard_weights, tau=0.01)
        hard_weights = hard_weights[:, 1].view(-1, self.n_agents, 1,self.n_agents - 1)
        hard_weights = hard_weights.permute(1, 0, 2, 3)

        # Soft Attention
        q = self.q(h_out).reshape(-1,self.n_agents,self.attention_dim)  # (batch_size, n_agents, args.attention_dim)
        k = self.k(h_out).reshape(-1,self.n_agents,self.attention_dim)  # (batch_size, n_agents, args.attention_dim)

        x = []
        for i in range(self.n_agents):
            q_i = q[:, i].view(-1, 1, self.attention_dim)
            k_i = [k[:, j] for j in range(self.n_agents) if j != i]

            k_i = torch.stack(k_i, dim=0)
            k_i = k_i.permute(1, 2, 0)
            score = torch.matmul(q_i, k_i)
            scaled_score = score / np.sqrt(self.attention_dim)
            soft_weight = f.softmax(scaled_score, dim=-1)  # (batch_size，1, n_agents - 1)
            x.append(soft_weight[0][0] * hard_weights[i][0][0])

        xx = torch.stack(x, dim=0)
        obs = obs.reshape(3, -1)
        obs = obs.numpy()
        ran = [0, 1, 2]
        res = []
        for i in range(len(obs)):
            ran_c = [ran[m] for m in range(len(obs)) if m != i]
            for j in range(len(obs) - 1):
                fla = 0
                v = ran_c[j]
                for k in range(42):
                    fla = fla + (obs[i][k] - obs[v][k]) * (obs[i][k] - obs[v][k])
                res.append(fla)
        res_arr = np.array(res).reshape(len(obs), len(obs)-1)
        res_arr = torch.Tensor(res_arr)

        temp = []
        for i in range(len(res_arr)):
            j = torch.argmax(res_arr[i])
            if res_arr[i][j] != 0:
                if j >= i:
                    j += 1
            t = [i, j.item()]
            t.sort()
            temp.append(t)

        flag = 0
        for i in range(len(temp)):
            for j in range(len(temp)):
                if j > i:
                    if temp[i] == temp[j]:
                        flag = temp[i]
                        break
        return  h_out, xx, flag