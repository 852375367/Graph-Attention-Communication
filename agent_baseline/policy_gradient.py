# from utils.rl_utils import *
import torch as th
import torch.optim as optim
from torch.distributions import Categorical


class PGLearner:
    def __init__(self, mac, args):
        self.args = args
        self.n_agents = args.n_agents
        self.action_discrete_one_hot_shape = args.action_discrete_one_hot_shape
        self.action_continuous_shape = args.action_continuous_shape
        self.mac = mac

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.params = list(self.mac.parameters())
        self.critic_params = list(self.mac.critic_parameters())

        self.optimiser = optim.Adam(self.params, lr=args.lr, eps=self.args.optim_eps)

        self.critic_optimiser = optim.Adam(self.critic_params, lr=args.critic_lr, eps=self.args.optim_eps)

        self.supervised_optimiser = optim.Adam(list(self.mac.supervised_parameters()), lr=args.supervised_lr, eps=self.args.optim_eps)

        if self.args.cuda:
            self.cuda()

    def train(self, batch, awacs_index, n_return):
        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]

        episode_num = batch['o'].shape[0]
        self.mac.init_hidden(episode_num)

        for key in batch.keys():
            batch[key] = th.tensor(batch[key], dtype=th.float32).to(self.args.device)

        obs, r, mask, terminated = batch['o'], batch['r'], batch['avail_u'], batch['terminated']
        v = batch['v']
        n_return = n_return[:, :max_episode_len].to(self.args.device)
        u_discrete_log_prob, u_continuous_log_prob = batch['u_discrete_log_prob'], batch['u_continuous_log_prob']
        u_discrete, u_continuous = batch['u_discrete'], batch['u_continuous']

        # n_return = self._get_returns(r, terminated, max_episode_len)
        discrete_log_prob, continuous_log_prob, supervised_action_discrete, supervised_action_continuous, qs, supervised_discrete_action_probs = [], [], [], [], [], []
        discrete_entropy, continuous_entropy = [], []

        is_missile = obs[:, :, :, 0].unsqueeze(dim=-1)
        enemy_continuous_act_true = batch['enemy_o'][:, :, :, awacs_index]

        enemy_continuous_action_true = []
        for i in range(max_episode_len):
            if i + 1 < max_episode_len:
                is_missile[:, i, :, 0] = is_missile[:, i, :, 0] - is_missile[:, i + 1, :, 0]
            mac_input = obs[:, i, :, :]
            mask_ = mask[:, i, :, :]
            mask_ = list(th.split(mask_, self.args.discrete_shape, dim=-1))
            discrete_prob_, _, _, supervised_act, qs_, supervised_discrete_act_probs, _, continuous_distribution = self.mac.forward(mac_input, mask_, False, batch['enemy_o'][:, i, :, :])
            # continuous_entropy_ = continuous_distribution.entropy().reshape(self.args.batch_size, self.n_agents, -1)
            # continuous_entropy.append(continuous_entropy_)
            dis_entropy, dis_log_prob = [], []
            for j, dis_prob_ in enumerate(discrete_prob_):
                dist = Categorical(dis_prob_)
                dis_entropy.append(dist.entropy().unsqueeze(dim=-1))
                dis_log_prob.append(dist.log_prob(u_discrete[:, i, :, j]).unsqueeze(dim=-1))

            discrete_log_prob.append(th.cat(dis_log_prob, dim=-1))
            discrete_entropy.append(th.cat(dis_entropy, dim=-1))
            continuous_log_prob_ = continuous_distribution.log_prob(u_continuous[:, i, :, :].reshape(-1, self.args.action_continuous_shape)).reshape(self.args.batch_size, self.n_agents, -1)
            continuous_log_prob.append(continuous_log_prob_)

            supervised_act_discrete, supervised_act_continuous, supervised_continuous_scale = supervised_act
            supervised_action_continuous.append(supervised_act_continuous.reshape(episode_num, self.n_agents, -1))
            qs.append(qs_)
            supervised_discrete_action_probs.append(supervised_discrete_act_probs)
            enemy_continuous_act_true_ = (enemy_continuous_act_true[:, i, :, :]).reshape(episode_num * self.n_agents, -1)
            supervised_continuous_1 = th.gather(enemy_continuous_act_true_, 1, th.tensor([[0, 1, 2] for _ in range(episode_num * self.n_agents)]).to(self.args.device)) / 525.0
            supervised_continuous_2 = th.gather(enemy_continuous_act_true_, 1, th.tensor([[3, 4] for _ in range(episode_num * self.n_agents)]).to(self.args.device)) / 25000.0
            supervised_continuous_3 = (th.gather(enemy_continuous_act_true_, 1, th.tensor([[5] for _ in range(episode_num * self.n_agents)]).to(self.args.device)) + 10000) / 10000.0
            enemy_continuous_action_true.append(th.cat((supervised_continuous_1, supervised_continuous_2, supervised_continuous_3), dim=-1).reshape(episode_num, self.n_agents, -1))

        is_missile[:, -1:, :, :] = 0
        is_missile_true = th.zeros(is_missile.shape[0], is_missile.shape[1], is_missile.shape[2], is_missile.shape[3] + 1).to(self.args.device).scatter_(3, ((is_missile.long()).bool()).long(), 1)

        discrete_log_prob = th.stack(discrete_log_prob, dim=1)
        continuous_log_prob = th.stack(continuous_log_prob, dim=1)
        discrete_entropy = th.stack(discrete_entropy, dim=1)
        qs = th.stack(qs, dim=1)

        supervised_action_continuous = th.stack(supervised_action_continuous, dim=1)
        supervised_discrete_action_probs = th.stack(supervised_discrete_action_probs, dim=1)
        enemy_continuous_action_true = th.stack(enemy_continuous_action_true, dim=1)

        adv_targ = (n_return - v.squeeze()).unsqueeze(dim=-1)
        adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-5)

        critic_loss = self.args.value_loss_coef * ((n_return - qs.squeeze()) ** 2).mean()
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        discrete_ratio = th.exp(discrete_log_prob - u_discrete_log_prob)
        discrete_surr1 = (discrete_ratio * adv_targ.detach())
        discrete_surr2 = (th.clamp(discrete_ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * adv_targ.detach())
        discrete_loss = - th.min(discrete_surr1, discrete_surr2).mean()

        continuous_ratio = th.exp(continuous_log_prob - u_continuous_log_prob)
        continuous_surr1 = continuous_ratio * adv_targ.detach()
        continuous_surr2 = th.clamp(continuous_ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * adv_targ.detach()
        continuous_loss = - th.min(continuous_surr1, continuous_surr2).mean()

        actor_loss = discrete_loss + continuous_loss
        # entropy_loss = - self.args.entropy_coef * (discrete_entropy.mean() + continuous_entropy.mean())
        entropy_loss = - self.args.entropy_coef * discrete_entropy.mean()
        loss = actor_loss + entropy_loss

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.train_supervised(enemy_continuous_action_true, supervised_discrete_action_probs, supervised_action_continuous, is_missile_true)


    def train_supervised(self, enemy_continuous_action_true, supervised_discrete, supervised_continuous, is_missile):
        continuous_action_loss = (th.square(enemy_continuous_action_true - supervised_continuous)).mean()
        out1 = list(th.split(supervised_discrete, [supervised_discrete.shape[-1] // 3, supervised_discrete.shape[-1] // 3, supervised_discrete.shape[-1] // 3], dim=-1))
        discrete_action_loss = 0

        for i in range(self.args.n_agents):
            enemy_is_misslie = ((is_missile[:, :, i, :]).unsqueeze(dim=-2).expand(out1[i].shape[0], out1[i].shape[1], self.n_agents, out1[i].shape[-1]))
            discrete_action_loss += (enemy_is_misslie * th.log(out1[i] + 1e-8)).mean()

        loss = continuous_action_loss - discrete_action_loss
        self.supervised_optimiser.zero_grad()
        loss.backward()
        self.supervised_optimiser.step()

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len


    def cuda(self):
        self.mac.cuda()

    def save_models(self, path, episode, side):
        self.mac.save_models(path, episode, side)
        th.save(self.optimiser.state_dict(), "{0}/opt_{1}_{2}.th".format(path, episode, side))

    def load_models(self, path, episode, side):
        self.mac.load_models(path, episode, side)
        self.optimiser.load_state_dict(th.load("{0}/opt_{1}_{2}.th".format(path, episode, side), map_location=lambda storage, loc: storage))

