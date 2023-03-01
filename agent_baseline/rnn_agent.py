import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.in_fn = nn.BatchNorm1d(input_shape*2)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)

        self.in_fn2 = nn.BatchNorm1d(args.rnn_hidden_dim)
        self.in_fn2.weight.data.fill_(1)
        self.in_fn2.bias.data.fill_(0)

        self.fc1 = nn.Linear(input_shape*2, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.supervised_action_shape_0)

        self.fc1_ = nn.Linear(input_shape*2 * 3, args.rnn_hidden_dim)
        self.rnn_ = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.supervised_action_shape_1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_(), self.fc1_.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, hidden_state_):
        x = self.in_fn(inputs)
        x1 = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x1, h_in)
        out1 = self.fc2(self.in_fn2(h))
        batch_size = x.shape[0] // self.args.n_agents
        x_ = x.reshape(batch_size, -1).unsqueeze(dim=1)
        x_ = x_.expand(batch_size, self.args.n_agents, -1).reshape(batch_size * self.args.n_agents, -1)
        x2 = F.relu(self.fc1_(x_))
        h_in_ = hidden_state_.reshape(-1, self.args.rnn_hidden_dim)
        h_ = self.rnn(x2, h_in_)
        out2 = th.tanh(self.fc3(self.in_fn2(h_)))
        return out1, out2, h, h_

class RNNPPOAgent(nn.Module):

    def __init__(self, input_shape, args):
        super(RNNPPOAgent, self).__init__()
        self.args = args

        self.in_fn = nn.BatchNorm1d(input_shape*2)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)
        self.fc1 = nn.Linear(input_shape*2, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # self.fc_discrete_action = nn.Linear(args.rnn_hidden_dim, args.action_discrete_one_hot_shape)
        self.fc_discrete_action = nn.ModuleList()
        for i in range(len(args.discrete_shape)):
            self.fc_discrete_action.append(nn.Linear(args.rnn_hidden_dim, args.discrete_shape[i]))
        self.fc_continuous_mu = nn.Linear(args.rnn_hidden_dim, args.action_continuous_shape)
        # self.fc_continuous_sigma = nn.Linear(args.rnn_hidden_dim, args.action_continuous_shape)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = self.in_fn(inputs)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        discrete_action = []
        for i, fc in enumerate(self.fc_discrete_action):
            discrete_action.append(fc(h))

        continuous_mu = th.tanh(self.fc_continuous_mu(h))

        return discrete_action, continuous_mu, h


class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        # input_shape = (obs_shape + len(self.args.discrete_shape) + self.args.action_continuous_shape) * self.args.n_agents
        # input_shape = obs_shape * self.args.n_agents
        self.in_fn = nn.BatchNorm1d(input_shape + self.args.obs_shape * 3)
        self.in_fn.weight.data.fill_(1)
        self.in_fn.bias.data.fill_(0)
        self.fc1 = nn.Linear(input_shape + self.args.obs_shape * 3, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_q = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = self.in_fn(inputs)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc_q(h)
        return q, h
