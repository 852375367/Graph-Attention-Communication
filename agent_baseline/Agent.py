
from agent_baseline.policy_gradient import PGLearner as Learner
from agent_baseline.ppo_controller import PPOMAC as Controller
from agent_baseline.replay_buffer import ReplayBuffer
import numpy as np
import torch as th
import math
import os
from pathlib import Path
import argparse
from framwork.agent_base import AgentBase


class Agent(AgentBase):
    def __init__(self, config):
        super(Agent, self).__init__()
        # self.args = self.get_args()
        self.args = config.args
        self.t_env = 0
        self.evaluate = self.args.test_mode  # evaluate == True, 评估模型；evaluate==False， 训练模型
        self.win_flag = []
        self.episode_rewards = []
        self.episode_limit = self.args.episode_limit
        self.current_episode = 0
        if self.evaluate:
            self.model_id = self.args.incremental
            self.model_dir = str(Path(__file__).resolve().parent)
        else:
            self.model_dir = Path(str(Path(__file__).resolve().parent / 'models'))
            if not self.model_dir.exists():
                curr_run = 'run1'
            else:
                exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in self.model_dir.iterdir()
                                 if str(folder.name).startswith('run')]
                if len(exst_run_nums) == 0:
                    curr_run = 'run1'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)

            self.run_dir = self.model_dir / curr_run
            self.results_dir = self.run_dir / 'results'
            os.makedirs(str(self.results_dir))

        th.manual_seed(self.args.seed)  # seed set up as 1
        np.random.seed(self.args.seed)

        self.unnecessary_state_index = [26, 27, 28, 29, 30, 31, 54, 55, 56, 57, 58, 59, 82, 83, 84, 85, 86, 87, 110, 111,
                                   112, 113, 114, 115, 131, 133, 135, 137, 139, 141, 160, 161, 162, 165, 166, 167, 169,
                                   170, 171, 201, 202, 203]

        self.init_component = 0
        self.args.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.device = self.args.device

    def before_reset(self, env, side):
        pass

    def after_reset(self, env, side):
        self.env = env
        if side == 'red':
            self.side = 0
        else:
            self.side = 1
        self.blue_destroyed = []
        self.red_destroyed = []
        self.red_missile = [4, 4,4]
        self.blue_missile = [4, 4,4]
        self.step_reward = []
        self.step = 0
        self.o, self.r, self.u_discrete, self.u_continuous, self.u_discrete_prob, self.u_continuous_prob, self.avail_u, self.terminate = [], [], [], [], [], [], [], []
        self.enemy_o = []
        self.v = []
        if self.side == 0:
            self.awacs_ind = [640,641,642,643,644,645]
        else:
            self.awacs_ind = [1567, 1568, 1569, 1570, 1571, 1572]
        obs, awacs_index = self.build_state(self.side)
        self.obs = np.array(obs)
        self.awacs_index = awacs_index
        n_agents = self.obs.shape[0]
        obs_shape = self.obs.shape[1]
        discrete_shape, continuous_shape, supervised_action_shape_0, supervised_action_shape_1 = self.build_action()
        self.args.n_agents = n_agents
        self.args.obs_shape = obs_shape
        self.args.input_shape = obs_shape + n_agents
        self.args.discrete_shape = discrete_shape
        self.args.action_discrete_one_hot_shape = np.sum(discrete_shape)
        self.args.action_continuous_shape = continuous_shape
        self.args.supervised_action_shape_0 = supervised_action_shape_0
        self.args.supervised_action_shape_1 = supervised_action_shape_1
        self.action_discrete = len(self.args.discrete_shape)

        if self.init_component == 0:
            self.replay_buffer = ReplayBuffer(self.args)
            self.mac = Controller(self.args.input_shape, self.args)
            self.learner = Learner(self.mac, self.args)
            if self.evaluate:
                self.learner.load_models(str(self.model_dir), self.model_id, self.side)
            self.init_component += 1
        self.mac.init_hidden(1)

    def before_step_for_sample(self, env):
        self.obs, self.awacs_index = self.build_state(self.side)
        self.obs = np.expand_dims(np.array(self.obs), axis=0)
        self.enemy_obs, _ = self.build_state(1 - self.side)
        self.enemy_obs = np.expand_dims(np.array(self.enemy_obs), axis=0)
        self.avail_actions = self.build_action_mask(self.side)

        if self.evaluate:
            discrete_action, continuous_action, continuous_action_prob, supervised_action, supervised_discrete_action_probs = self.mac.select_actions(self.obs, self.avail_actions, self.t_env, self.evaluate)
        else:
            discrete_action, discrete_prob, continuous_action, continuous_action_prob, q, supervised_action, supervised_discrete_action_probs, continuous_distribution_action = self.mac.select_actions(self.obs, self.avail_actions, self.t_env, self.evaluate, self.enemy_obs)
            self.discrete_prob = discrete_prob.squeeze(dim=0)
            self.value = q.squeeze(dim=0)
            self.continuous_distribution_action = continuous_distribution_action.squeeze(dim=0)

        self.discrete_action = discrete_action.squeeze(dim=0)
        self.continuous_action = continuous_action.squeeze(dim=0)
        self.continuous_action_prob = continuous_action_prob.squeeze(dim=0)

        if self.side == 0:
            self.red_action(self.discrete_action.cpu().detach().numpy(), self.continuous_action.cpu().detach().numpy(), supervised_action)
        else:
            self.blue_action(self.discrete_action.cpu().detach().numpy(), self.continuous_action.cpu().detach().numpy(), supervised_action)

    def after_step_for_sample(self, env):
        if self.evaluate == False:
            reward = self.get_reward(self.side)
            terminated = self.env.done
            self.step_reward.append(np.sum(reward))
            self.step += 1
            if terminated:
                win_tag = self.env.judge_red_win()
                if self.side == 0:
                    self.win_flag.append(win_tag)
                else:
                    self.win_flag.append(0 - win_tag)
                self.t_env += self.step

            self.o.append(np.squeeze(self.obs))
            self.u_discrete.append(self.discrete_action.cpu().detach().numpy())
            self.u_continuous.append(self.continuous_distribution_action.cpu().detach().numpy())
            self.u_discrete_prob.append(self.discrete_prob.cpu().detach().numpy())
            self.u_continuous_prob.append(self.continuous_action_prob.cpu().detach().numpy())
            self.avail_actions = th.cat(self.avail_actions, dim=1)
            self.avail_u.append(self.avail_actions.cpu().detach().numpy())
            self.v.append(self.value.cpu().detach().numpy())
            self.r.append(reward)
            self.terminate.append([float(terminated)])
            self.enemy_o.append(np.squeeze(self.enemy_obs))

    def before_step_for_train(self, env):
        obs, awacs_index = self.build_state(self.side)
        obs = np.expand_dims(np.array(obs), axis=0)
        self.o.append(np.squeeze(obs))
        enemy_obs, _ = self.build_state(1 - self.side)
        enemy_obs = np.expand_dims(np.array(enemy_obs), axis=0)
        self.enemy_o.append(np.squeeze(enemy_obs))

        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = self.build_action_mask(self.side)
        avail_actions = th.cat(avail_actions, dim=1)
        self.avail_u.append(avail_actions.cpu().detach().numpy())

        batch_size = obs.shape[0]
        agent_inputs = th.tensor(obs, dtype=th.float32).to(self.device)
        agent_enemy_obs = th.tensor(enemy_obs, dtype=th.float32).to(self.device)
        agent_id_one_hot = th.eye(self.args.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        agent_inputs = th.cat([agent_inputs, agent_id_one_hot], dim=-1)
        critic_inputs = th.cat([agent_inputs, agent_enemy_obs], dim=-1)
        critic_inputs = critic_inputs.reshape(batch_size, -1).unsqueeze(dim=1).expand(-1, self.n_agents, -1).reshape(batch_size * self.n_agents, -1)
        last_q, self.critic_hidden_states = self.mac.critic(critic_inputs, self.mac.critic_hidden_states)
        self.v.append(last_q.squeeze(dim=0).cpu().detach().numpy())

        for i in range(self.step, self.episode_limit):
            self.o.append(np.zeros((self.n_agents, self.args.obs_shape)))
            self.r.append([0., 0., 0.])
            self.u_discrete.append(np.zeros([self.n_agents, self.action_discrete]))
            self.u_continuous.append(np.zeros([self.n_agents, self.args.action_continuous_shape]))
            self.u_discrete_prob.append(np.zeros((self.n_agents, self.action_discrete)))
            self.u_continuous_prob.append(np.zeros((self.n_agents, self.args.action_continuous_shape)))
            self.avail_u.append(np.zeros((self.n_agents, self.args.action_discrete_one_hot_shape)))
            self.terminate.append([1.])
            self.enemy_o.append(np.zeros((self.n_agents, self.args.obs_shape)))
            self.v.append(np.zeros((self.n_agents, 1)))

        if self.win_flag[-1] == 1:
            self.r[self.step - 1] = [5.0 + self.r[self.step - 1][0], 5.0 + self.r[self.step - 1][1], 5.0 + self.r[self.step - 1][2]]
        elif self.win_flag[-1] == -1:
            self.r[self.step - 1] = [-2.0 + self.r[self.step - 1][0], -2.0 + self.r[self.step - 1][1], -2.0 + self.r[self.step - 1][2]]

        episode = dict(o=self.o.copy(),
                       r=self.r.copy(),
                       avail_u=self.avail_u.copy(),
                       u_discrete=self.u_discrete.copy(),
                       u_continuous=self.u_continuous.copy(),
                       u_discrete_log_prob=self.u_discrete_prob.copy(),
                       u_continuous_log_prob=self.u_continuous_prob.copy(),
                       terminated=self.terminate.copy(),
                       enemy_o=self.enemy_o.copy(),
                       v=self.v.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]])

        self.replay_buffer.store_episode(episode)
        self.current_episode += 1
        self.episode_rewards.append(np.sum(self.step_reward))

    def get_batchs(self):
        mini_batch, n_return = self.replay_buffer.sample(min(self.replay_buffer.current_size, self.args.batch_size))
        self.n_returns = n_return
        return mini_batch

    def after_step_for_train(self, env):
        if self.current_episode % (self.args.save_interval) == 0:
            os.makedirs(str(self.run_dir / 'incremental'), exist_ok=True)
            dir = self.run_dir / 'incremental'
            self.learner.save_models(str(self.run_dir / 'incremental'), self.current_episode, self.side)
            self.learner.save_models(str(self.run_dir), 0, self.side)

    def train(self, batchs):
        self.learner.train(batchs, self.awacs_index, self.n_returns)

    def get_interval(self):
        return self.args.save_interval

    def print_train_log(self):
        pass

    def build_state(self, side):
        obs_n = []
        awacs_index = []
        self.n_agents = len(self.env.state_interface['AMS']) // 2
        if side == 0:
            states = self.env.state_interface['AMS'][:(self.n_agents)]
        else:
            states = self.env.state_interface['AMS'][-(self.n_agents):]

        for obs in states:
            obs_ = []
            val_ = []
            self.env.to_list(obs_, obs)
            i = 0

            for ind, val in enumerate(obs_):
                if ind in self.unnecessary_state_index:
                    continue
                if val['value_index'] in self.awacs_ind:
                    awacs_index.append(i)
                if math.isnan(val['value']):
                    val_.append(0.0)
                elif math.isinf(val["value"]):
                    val_.append(0.0)
                else:
                    action_true = 0
                    if val['value'] > val['max']:
                        action_true = val['max']
                    elif val['value'] < val['min']:
                        action_true = val['min']
                    else:
                        action_true = val['value']
                    val_.append(action_true)
                i += 1
            obs_n.append(val_)
        return obs_n, awacs_index

    def build_action(self):
        discrete_shape = []  # discrete one-hot维度
        continuous_shape = 0
        supervised_action_shape_0 = 0
        supervised_action_shape_1 = 0
        red_action = self.env.action_interface["AMS"][0]
        action_ = []
        self.env.to_list(action_, red_action)

        for shape in action_[:-1]:
            if shape['value_index'] == 59 or shape['value_index'] == 88:
                continue
            elif shape['value_index'] == 279 or shape['value_index'] == 282 or shape['value_index'] == 285:
                supervised_action_shape_0 += shape["mask_len"]
            elif shape["mask_len"] > 0:
                discrete_shape.append(shape["mask_len"])
            else:
                continuous_shape += 1
        discrete_shape[-1] = discrete_shape[-1] + 1
        red_action = self.env.action_interface["red_awacs"][0]
        action_ = []
        self.env.to_list(action_, red_action)
        supervised_action_shape_1 += len(action_)

        return discrete_shape, continuous_shape, supervised_action_shape_0, supervised_action_shape_1

    def build_action_mask(self, side):
        if side == 0:
            action_mask = []
            red_action = self.env.action_interface["AMS"][0]

            action_0 = []
            self.env.to_list(action_0, red_action)

            red_action = self.env.action_interface["AMS"][1]
            action_1 = []
            self.env.to_list(action_1, red_action)

            red_action = self.env.action_interface["AMS"][2]
            action_2 = []
            self.env.to_list(action_2, red_action)

            action_0[-2]["mask"] = [1.0, 1.0, 1.0, 1.0]
            action_1[-2]["mask"] = [1.0, 1.0, 1.0, 1.0]
            action_2[-2]["mask"] = [1.0, 1.0, 1.0, 1.0]

            continue_index = [279, 282, 285, 59, 88, 321, 348]
            for shape_0, shape_1 ,shape_2 in zip(action_0[:-1], action_1[:-1], action_2[:-1]):
                if shape_0['value_index'] in continue_index or shape_1['value_index']  in continue_index or shape_2['value_index']  in continue_index:
                    continue
                elif shape_0["mask_len"] > 0:
                    action_mask.append(th.tensor([shape_0["mask"], shape_1["mask"], shape_2["mask"]], dtype=th.float32).to(self.device))
            return action_mask
        else:
            blue_action_mask = []
            blue_action = self.env.action_interface["AMS"][3]

            action_0 = []
            self.env.to_list(action_0, blue_action)

            blue_action = self.env.action_interface["AMS"][4]
            action_1 = []
            self.env.to_list(action_1, blue_action)

            blue_action = self.env.action_interface["AMS"][5]
            action_2 = []
            self.env.to_list(action_2, blue_action)
            action_0[-2]["mask"] = [1.0, 1.0, 1.0, 1.0]
            action_1[-2]["mask"] = [1.0, 1.0, 1.0, 1.0]
            action_2[-2]["mask"] = [1.0, 1.0, 1.0, 1.0]
            continue_index = [1206, 1209, 1212, 1041, 1044, 986, 1015, 851, 878]

            for shape_0, shape_1, shape_2 in zip(action_0[:-1], action_1[:-1], action_2[:-1]):
                if shape_0['value_index'] in continue_index or shape_1['value_index'] in continue_index or shape_2['value_index'] in continue_index:
                    continue
                elif shape_0["mask_len"] > 0:
                    blue_action_mask.append(th.tensor([shape_0["mask"], shape_1["mask"], shape_2["mask"]], dtype=th.float32).to(self.device))
            return blue_action_mask

    def red_action(self, discrete_action, continuous_action, supervised_action):
        supervised_discrete, supervised_continuous, supervised_continuous_scale = supervised_action[0].cpu().detach().numpy(), supervised_action[1].cpu().detach().numpy(), supervised_action[2].cpu().detach().numpy()

        if "red_awacs" in self.env.action_interface.keys():
            for i in range(self.env.blue):
                self.env.action_interface["red_awacs"][i]["action_xg_0_est"]["value"] = supervised_continuous_scale[i][3]
                self.env.action_interface["red_awacs"][i]["action_xg_1_est"]["value"] = supervised_continuous_scale[i][4]
                self.env.action_interface["red_awacs"][i]["action_xg_2_est"]["value"] = supervised_continuous_scale[i][5]
                self.env.action_interface["red_awacs"][i]["action_vg_0_est"]["value"] = supervised_continuous_scale[i][0]
                self.env.action_interface["red_awacs"][i]["action_vg_1_est"]["value"] = supervised_continuous_scale[i][1]
                self.env.action_interface["red_awacs"][i]["action_vg_2_est"]["value"] = supervised_continuous_scale[i][2]

        for i, action in enumerate(self.env.action_interface["AMS"]):
            if i < self.env.red:  # red
                action["SemanticManeuver"]["clockwise_cmd"]["value"] = discrete_action[i][0]
                action["SemanticManeuver"]["combat_mode"]["value"] = 0
                action["SemanticManeuver"]["flag_after_burning"]["value"] = 1
                action["SemanticManeuver"]["horizontal_cmd"]["value"] = discrete_action[i][1]
                action["SemanticManeuver"]["maneuver_target"]["value"] = discrete_action[i][2]
                action["SemanticManeuver"]["ny_cmd"]["value"] = continuous_action[i][0]
                action["SemanticManeuver"]["vel_cmd"]["value"] = continuous_action[i][1]
                action["SemanticManeuver"]["vertical_cmd"]["value"] = discrete_action[i][3]
                action["action_shoot_predict_list"][0]["shoot_predict"]["value"] = supervised_discrete[0][i][0]
                action["action_shoot_predict_list"][1]["shoot_predict"]["value"] = supervised_discrete[0][i][1]
                action["action_shoot_predict_list"][2]["shoot_predict"]["value"] = supervised_discrete[0][i][2]
                action["action_shoot_target"]["value"] = discrete_action[i][4] - 1  # random.randint(-1,1)
                action["action_target"]["value"] = discrete_action[i][2]

    def blue_action(self, discrete_action, continuous_action, supervised_action):

        supervised_discrete, supervised_continuous, supervised_continuous_scale = supervised_action[0].cpu().detach().numpy(), supervised_action[1].cpu().detach().numpy(), supervised_action[2].cpu().detach().numpy()

        if "blue_awacs" in self.env.action_interface.keys():
            for i in range(self.env.blue):
                self.env.action_interface["blue_awacs"][i]["action_xg_0_est"]["value"] = supervised_continuous_scale[i][3]
                self.env.action_interface["blue_awacs"][i]["action_xg_1_est"]["value"] = supervised_continuous_scale[i][4]
                self.env.action_interface["blue_awacs"][i]["action_xg_2_est"]["value"] = supervised_continuous_scale[i][5]
                self.env.action_interface["blue_awacs"][i]["action_vg_0_est"]["value"] = supervised_continuous_scale[i][0]
                self.env.action_interface["blue_awacs"][i]["action_vg_1_est"]["value"] = supervised_continuous_scale[i][1]
                self.env.action_interface["blue_awacs"][i]["action_vg_2_est"]["value"] = supervised_continuous_scale[i][2]

        for i, action in enumerate(self.env.action_interface["AMS"]):
            if i >= self.env.red:
                i -= 3
                action["SemanticManeuver"]["clockwise_cmd"]["value"] = discrete_action[i][0]
                action["SemanticManeuver"]["combat_mode"]["value"] = 0
                action["SemanticManeuver"]["flag_after_burning"]["value"] = 1
                action["SemanticManeuver"]["horizontal_cmd"]["value"] = discrete_action[i][1]
                action["SemanticManeuver"]["maneuver_target"]["value"] = discrete_action[i][2]
                action["SemanticManeuver"]["ny_cmd"]["value"] = continuous_action[i][0]
                action["SemanticManeuver"]["vel_cmd"]["value"] = continuous_action[i][1]
                action["SemanticManeuver"]["vertical_cmd"]["value"] = discrete_action[i][3]
                action["action_shoot_predict_list"][0]["shoot_predict"]["value"] = supervised_discrete[0][i][0]
                action["action_shoot_predict_list"][1]["shoot_predict"]["value"] = supervised_discrete[0][i][1]
                action["action_shoot_predict_list"][2]["shoot_predict"]["value"] = supervised_discrete[0][i][2]
                action["action_shoot_target"]["value"] = discrete_action[i][4] - 1  # random.randint(-1,1)
                action["action_target"]["value"] = discrete_action[i][2]

    def get_reward(self, side):
        red_missile = 0
        blue_missile = 0
        red_reward = np.array([0, 0, 0], dtype=np.float32)
        blue_reward = np.array([0, 0, 0], dtype=np.float32)
        # Counting red/blue alive agents
        for i in range(self.env.red):
            if self.env.state_interface["AMS"][i]["alive"]["value"] < 1.0 and (i not in self.red_destroyed):
                self.red_destroyed.append(i)
                red_reward[i] -= 3
                blue_reward += 2
            red_missile += self.red_missile[i] - self.env.state_interface["AMS"][i]["AAM_remain"]["value"]
            red_reward[i] = red_reward[i] - red_missile * 0.01
            self.red_missile[i] = self.env.state_interface["AMS"][i]["AAM_remain"]["value"]
        for i in range(self.env.red, self.env.red + self.env.blue):
            if self.env.state_interface["AMS"][i]["alive"]["value"] < 1.0 and (i not in self.blue_destroyed):
                self.blue_destroyed.append(i)
                blue_reward[i - self.env.red] -= 3
                red_reward += 2
            blue_missile += self.blue_missile[i - self.env.red] - self.env.state_interface["AMS"][i]["AAM_remain"]["value"]
            blue_reward[i - self.env.red] = blue_reward[i - self.env.red] - blue_missile * 0.01
            self.blue_missile[i - self.env.red] = self.env.state_interface["AMS"][i]["AAM_remain"]["value"]
        if side == 0:
            return red_reward.tolist()
        else:
            return blue_reward.tolist()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", default=1, type=int, help="Random seed")
        parser.add_argument("--n_episodes", default=1, type=int)
        parser.add_argument("--train_steps", default=16, type=int)
        parser.add_argument("--buffer_size", default=64, type=int)
        parser.add_argument("--t_max", default=1000000, type=int)
        parser.add_argument("--episode_limit", default=50, type=int)
        parser.add_argument("--batch_size", default=32, type=int, help="Batch size for model training")

        parser.add_argument("--epsilon_start", default=1.0, type=float)
        parser.add_argument("--epsilon_finish", default=0.01, type=float)
        parser.add_argument("--epsilon_anneal_time", default=50000, type=float)
        parser.add_argument("--save_interval", default=100, type=int)

        parser.add_argument("--hidden_dim", default=64, type=int)
        parser.add_argument("--rnn_hidden_dim", default=64, type=int)

        parser.add_argument("--value_loss_coef", default=0.5, type=float)
        parser.add_argument("--entropy_coef", default=0.05, type=float)
        parser.add_argument("--lr", default=0.0003, type=float)
        parser.add_argument("--critic_lr", default=0.0003, type=float)
        parser.add_argument("--supervised_lr", default=0.001, type=float)
        parser.add_argument("--optim_alpha", default=0.99, type=float)
        parser.add_argument("--optim_eps", default=0.00001, type=float)
        parser.add_argument("--std_noise", default=0.25, type=int)
        parser.add_argument("--gamma", default=0.99, type=float)
        parser.add_argument("--clip_param", default=0.2, type=float)
        parser.add_argument("--gae", default=True, type=bool)
        parser.add_argument("--gae_lambda", default=0.95, type=float)

        parser.add_argument("--test_greedy", default=True, type=bool)
        parser.add_argument("--cuda", default=False, type=bool)

        parser.add_argument("--test_mode", default=False, type=bool, help="evaluate")
        parser.add_argument("--incremental", default=0, type=int)

        args = parser.parse_args()
        return args
