import numpy as np
import threading
import copy
import torch as th
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_agents = self.args.n_agents
        self.obs_shape = self.args.obs_shape
        self.action_discrete = len(args.discrete_shape)
        self.action_discrete_one_hot_shape = self.args.action_discrete_one_hot_shape
        self.action_continuous_shape = self.args.action_continuous_shape

        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        self.current_idx_predict = 0
        self.current_size_predict = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit + 1, self.n_agents, self.obs_shape]),
                        'r': np.empty([self.size, self.episode_limit, self.n_agents]),
                        'u_discrete': np.empty([self.size, self.episode_limit, self.n_agents, self.action_discrete]),
                        'u_continuous': np.empty([self.size, self.episode_limit, self.n_agents, self.action_continuous_shape]),
                        'avail_u': np.empty([self.size, self.episode_limit + 1, self.n_agents, self.action_discrete_one_hot_shape]),
                        'u_discrete_log_prob': np.empty([self.size, self.episode_limit, self.n_agents, self.action_discrete]),
                        'u_continuous_log_prob': np.empty([self.size, self.episode_limit, self.n_agents, self.action_continuous_shape]),
                        'terminated': np.empty([self.size, self.episode_limit, 1]),
                        'v': np.empty([self.size, self.episode_limit + 1, self.n_agents, 1]),
                        'enemy_o': np.empty([self.size, self.episode_limit + 1, self.n_agents, self.obs_shape])
                        }
        self.returns = th.zeros([self.size, self.episode_limit, self.n_agents])
        self.blue_buffers = copy.deepcopy(self.buffers)

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['v'][idxs] = episode_batch['v']
            self.buffers['u_discrete'][idxs] = episode_batch['u_discrete']
            self.buffers['u_continuous'][idxs] = episode_batch['u_continuous']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['u_discrete_log_prob'][idxs] = episode_batch['u_discrete_log_prob']
            self.buffers['u_continuous_log_prob'][idxs] = episode_batch['u_continuous_log_prob']
            self.buffers['terminated'][idxs] = episode_batch['terminated']
            self.buffers['enemy_o'][idxs] = episode_batch['enemy_o']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer, self.returns[idx]

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def compute_returns(self):
        terminated = np.repeat(self.buffers['terminated'], self.n_agents, axis=-1)
        terminated = 1 - terminated
        max_episode_len = self.args.episode_limit
        r = th.tensor(self.buffers['r'], dtype=th.float32)
        self.returns[:, -1, :] = r[:, -1, :]
        v = self.buffers['v']
        if self.args.gae:
            gae = 0
            val = v.squeeze()
            for step in range(max_episode_len - 2, -1, -1):
                delta = r[:, step, :] + self.args.gamma * val[:, step + 1, :] * terminated[:, step, :] - val[:, step, :]
                gae = delta + self.args.gamma * self.args.gae_lambda * gae * terminated[:, step, :]
                self.returns[:, step, :] = gae + val[:, step, :]
        else:
            for step in range(max_episode_len - 2, -1, -1):
                self.returns[:, step, :] = r[:, step, :] + self.args.gamma * self.returns[:, step + 1, :] * terminated[:, step, :]

    # def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
    #     num_processes, num_steps = self.size, self.episode_limit
    #     batch_size = num_processes * num_steps
    #
    #     if mini_batch_size is None:
    #         assert batch_size >= num_mini_batch, (
    #             "PPO requires the number of processes ({}) "
    #             "* number of steps ({}) = {} "
    #             "to be greater than or equal to the number of PPO mini batches ({})."
    #             "".format(num_processes, num_steps, num_processes * num_steps,
    #                       num_mini_batch))
    #         mini_batch_size = batch_size // num_mini_batch
    #     sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)
    #     for indices in sampler:
    #         o_batch = self.buffers['o'][:-1].view(-1, *self.buffers['o'].size()[2:])[indices]
    #         u_discrete_batch = self.buffers['u_discrete'][:-1].view(-1, *self.buffers['u_discrete'].size()[2:])[indices]
    #         u_continuous_batch = self.buffers['u_continuous'][:-1].view(-1, *self.buffers['u_continuous'].size()[2:])[indices]
    #         avail_u_batch = self.buffers['avail_u'][:-1].view(-1, *self.buffers['avail_u'].size()[2:])[indices]
    #         u_discrete_log_prob_batch = self.buffers['u_discrete_log_prob'][:-1].view(-1, *self.buffers['u_discrete_log_prob'].size()[2:])[indices]
    #         u_continuous_log_prob_batch = self.buffers['u_continuous_log_prob'][:-1].view(-1, *self.buffers['u_continuous_log_prob'].size()[2:])[indices]
    #         return_batch = self.returns[:-1].view(-1, self.n_agents)[indices]
    #         v_batch = self.buffers['v'][:-1].view(-1, *self.buffers['v'].size()[2:])[indices]
    #         enemy_o_batch = self.buffers['enemy_o'][:-1].view(-1, *self.buffers['enemy_o'].size()[2:])[indices]
    #         if advantages is None:
    #             adv_targ = None
    #         else:
    #             adv_targ = advantages.view(-1, self.n_agents)[indices]
    #
    #         yield o_batch, u_discrete_batch, avail_u_batch, u_discrete_log_prob_batch, u_continuous_log_prob_batch, return_batch, v_batch, enemy_o_batch, adv_targ

