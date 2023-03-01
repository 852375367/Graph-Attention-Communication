import torch
import torch.nn as nn
from agent_baseline.g2a import G2A
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class G2atrain(nn.Module):
    BATCH_SIZE = 1
    LR = 0.00024

    def __init__(self,input_shape, args):
        super(G2atrain, self).__init__()
        self.AT=G2A(input_shape, args)
        self.n_agents = args.n_agents
        self.optimizer = torch.optim.Adam(self.AT.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

    def learn(self,hidden1,g2astates):

        hiddens, attention,flag = self.AT.forward(g2astates, hidden1)

        hiddens_1 = hiddens.detach().numpy()
        res = []
        for i in range(self.n_agents):
            temp = []
            for j in range(self.n_agents):
                if i == j:
                    continue
                temp.append(np.linalg.norm(hiddens_1[i] - hiddens_1[j]))
            res.append(self.softmax1(temp))
        res = torch.FloatTensor(res)
        loss = self.loss_func(res, attention)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def softmax1(self,v: [int]):
        l1 = list(map(lambda x: np.exp(x), v))
        return list(map(lambda x: x / sum(l1), l1))