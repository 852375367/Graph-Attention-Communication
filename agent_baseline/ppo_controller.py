from agent_baseline.rnn_agent import RNNPPOAgent
from agent_baseline.rnn_agent import RNNAgent
from agent_baseline.rnn_agent import Critic
from agent_baseline.action_selectors import MultinomialActionSelector as Action_selector
import torch as th
import torch.nn as nn
from agent_baseline.g2atrain import G2atrain


class PPOMAC:
    def __init__(self, input_shape, args):
        self.n_agents = args.n_agents
        self.args = args
        self._build_agents(input_shape)
        self.action_selector = Action_selector(args)
        self.hidden_states = None

    def select_actions(self, obs, avail_actions, t_env, test_mode=False, batch_enemy_obs=None):
        # Only select actions for the selected batch elements in bs

        discrete_action, continuous_action, continuous_action_prob, supervised_action, q, supervised_discrete_action_probs, continuous_distribution_action, _ = self.forward(obs, avail_actions, test_mode=test_mode, batch_enemy_obs=batch_enemy_obs)

        if test_mode:
            discrete_action, _ = self.action_selector.select_action(discrete_action, avail_actions, t_env, test_mode=test_mode)
            discrete_action = th.cat(discrete_action, dim=-1)
            return discrete_action, continuous_action, continuous_action_prob, supervised_action, supervised_discrete_action_probs
        else:
            discrete_action, discrete_action_log_probs = self.action_selector.select_action(discrete_action, avail_actions, t_env, test_mode=test_mode)
            discrete_action = th.cat(discrete_action, dim=-1)
            discrete_action_log_probs = th.cat(discrete_action_log_probs, dim=-1)
            return discrete_action, discrete_action_log_probs, continuous_action, continuous_action_prob, q, supervised_action, supervised_discrete_action_probs, continuous_distribution_action

    def forward(self, obs, avail_actions, test_mode=False, batch_enemy_obs=None):


        if not th.is_tensor(obs):
            inputs_g2a = th.tensor(obs, dtype=th.float32).to(self.args.device)
        else:
            inputs_g2a = obs
        agent_inputs_g2a=inputs_g2a[0]
        self.g2a_hidden_states,attention,flag =self.g2a.AT.forward(agent_inputs_g2a,self.g2a_hidden_states)
        self.g2a.learn(self.g2a_hidden_states.detach(), agent_inputs_g2a)

        batch_size = obs.shape[0]
        inputs = obs
        if not th.is_tensor(inputs):
            agent_inputs = th.tensor(inputs, dtype=th.float32).to(self.args.device)
        else:
            agent_inputs = inputs
        agent_id_one_hot = th.eye(self.args.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.args.device)
        agent_inputs = th.cat([agent_inputs, agent_id_one_hot], dim=-1)
        agent_inputs1=th.zeros(agent_inputs.shape).to(self.args.device)

        if flag!=0:
            agent_inputs1[0][0] = agent_inputs[0][1]
            agent_inputs1[0][1] = agent_inputs[0][0]

        agent_inputs_o=th.cat([agent_inputs, agent_inputs1], dim=-1)

        mac_inputs = agent_inputs_o.reshape(batch_size*self.n_agents, -1)

        discrete_action, continuous_mu, self.hidden_states = self.agent(mac_inputs, self.hidden_states)

        # get continuous action
        continuous_sigma = th.zeros(self.args.action_continuous_shape).to(self.args.device) + self.args.std_noise
        continuous_distribution = th.distributions.Normal(continuous_mu, continuous_sigma)
        continuous_distribution_action = continuous_distribution.sample()
        continuous_distribution_prob = continuous_distribution.log_prob(continuous_distribution_action).view(batch_size, self.n_agents, -1)
        # 0-9, 250-900
        continuous_action_0 = th.clamp(th.gather(continuous_distribution_action, 1, th.tensor([[0] for _ in range(batch_size*self.n_agents)]).to(self.args.device)), -1.0, 1.0) * 4.5 + 4.5
        continuous_action_1 = th.clamp(th.gather(continuous_distribution_action, 1, th.tensor([[1] for _ in range(batch_size*self.n_agents)]).to(self.args.device)), -1.0, 1.0) * 325 + 575
        continuous_action = th.cat((continuous_action_0, continuous_action_1), dim=1).view(batch_size, self.n_agents, -1)

        discrete_actions = discrete_action
        for i, (acts, avail_acts) in enumerate(zip(discrete_actions, avail_actions)):
            discrete_actions_i = th.nn.functional.softmax(acts, dim=-1)
            # if not test_mode:
            #     epsilon_action_num = discrete_actions[i].shape[-1]
            #     discrete_actions_i = ((1 - self.action_selector.epsilon) * discrete_actions_i) + th.ones_like(discrete_actions_i) * self.action_selector.epsilon / epsilon_action_num
            discrete_actions[i] = discrete_actions_i.view(batch_size, self.n_agents, -1)

        # gain supervised action
        out1, out2, self.supervised_hidden_states, self.supervised_hidden_states2 = self.supervised_agent(mac_inputs, self.supervised_hidden_states, self.supervised_hidden_states2)
        out1 = list(th.split(out1, [out1.shape[1] // 3, out1.shape[1] // 3, out1.shape[1] // 3], dim=1))
        supervised_out1 = []
        supervised_discrete_action_probs = []
        for i, acts in enumerate(out1):
            out1[i] = th.nn.functional.softmax(acts, dim=-1).view(batch_size, self.n_agents, -1)
            supervised_out1.append((out1[i].max(dim=-1)[1]).unsqueeze(dim=-1))
            supervised_discrete_action_probs.append(out1[i])
        supervised_out1 = th.cat(supervised_out1, dim=-1).float()
        supervised_discrete_action_probs = th.cat(supervised_discrete_action_probs, dim=-1).float()
        #  0-525(3), -50000-50000(2), -20000~-10000
        supervised_continuous_1 = th.clamp(th.gather(out2, 1, th.tensor([[0, 1, 2] for _ in range(batch_size*self.n_agents)]).to(self.args.device)), -1.0, 1.0)
        supervised_continuous_1_scale = supervised_continuous_1 * 525.0
        supervised_continuous_2 = th.clamp(th.gather(out2, 1, th.tensor([[3, 4] for _ in range(batch_size*self.n_agents)]).to(self.args.device)), -1.0, 1.0)
        supervised_continuous_2_scale = supervised_continuous_2 * 25000.0
        supervised_continuous_3 = th.clamp(th.gather(out2, 1, th.tensor([[5] for _ in range(batch_size*self.n_agents)]).to(self.args.device)), -1.0, 1.0)
        supervised_continuous_3_scale = supervised_continuous_3 * 10000 - 10000
        supervised_continuous = th.cat((supervised_continuous_1, supervised_continuous_2, supervised_continuous_3), dim=1)
        supervised_continuous_scale = th.cat((supervised_continuous_1_scale, supervised_continuous_2_scale, supervised_continuous_3_scale), dim=1)

        supervised_action = [supervised_out1, supervised_continuous, supervised_continuous_scale]
        if not test_mode:
            if not th.is_tensor(batch_enemy_obs):
                enemy_obs = th.tensor(batch_enemy_obs, dtype=th.float32).to(self.args.device)
            else:
                enemy_obs = batch_enemy_obs
            critic_inputs = th.cat([agent_inputs, enemy_obs], dim=-1)
            critic_inputs = critic_inputs.reshape(batch_size, -1).unsqueeze(dim=1).expand(-1, self.n_agents, -1).reshape(batch_size*self.n_agents, -1)
            q, self.critic_hidden_states = self.critic(critic_inputs, self.critic_hidden_states)
            return discrete_actions, continuous_action, continuous_distribution_prob, supervised_action, q.view(batch_size, self.n_agents, -1), supervised_discrete_action_probs, continuous_distribution_action.view(batch_size, self.n_agents, -1), continuous_distribution
        else:
            return discrete_actions, continuous_action, continuous_distribution_prob, supervised_action, None, supervised_discrete_action_probs, continuous_distribution_action.view(batch_size, self.n_agents, -1), continuous_distribution

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.critic_hidden_states = self.critic.init_hidden().expand(batch_size, self.n_agents, -1)
        self.supervised_hidden_states, self.supervised_hidden_states2 = self.supervised_agent.init_hidden()
        self.supervised_hidden_states = self.supervised_hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.supervised_hidden_states2 = self.supervised_hidden_states2.unsqueeze(0).expand(batch_size, self.n_agents, -1)

        self.g2a_hidden_states = self.g2a.AT.init_hidden()


    def parameters(self):
        return self.agent.parameters()

    def critic_parameters(self):
        return self.critic.parameters()

    def supervised_parameters(self):
        return self.supervised_agent.parameters()

    def cuda(self):
        self.agent.cuda()
        # for i in range(self.n_agents):
        #     self.agent[i].cuda()

        self.critic.cuda()
        self.supervised_agent.cuda()

    def save_models(self, path, episode, side):
        th.save(self.agent.state_dict(), "{0}/agent_{1}_{2}.th".format(path, episode, side))
        th.save(self.critic.state_dict(), "{0}/critic_{1}_{2}.th".format(path, episode, side))
        th.save(self.supervised_agent.state_dict(), "{0}/supervised_agent_{1}_{2}.th".format(path, episode, side))

    def load_models(self, path, episode, side):
        self.agent.load_state_dict(th.load("{0}/agent_{1}_{2}.th".format(path, episode, side), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(th.load("{0}/critic_{1}_{2}.th".format(path, episode, side), map_location=lambda storage, loc: storage))
        self.supervised_agent.load_state_dict(th.load("{0}/supervised_agent_{1}_{2}.th".format(path, episode, side), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = RNNPPOAgent(input_shape, self.args)
        self.critic = Critic(input_shape * self.args.n_agents, self.args)
        self.supervised_agent = RNNAgent(input_shape, self.args)
        self.g2a = G2atrain(input_shape, self.args)



