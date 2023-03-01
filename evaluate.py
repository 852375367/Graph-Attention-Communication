
import argparse
import torch
import os
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from environment.battlespace import BattleSpace as Environments
from agent_baseline.Agent import Agent
from agent_baseline.Agent import Agent as Agent_other
import copy

def run(args):

    env = Environments()
    agent_1 = Agent()
    agent_2 = Agent_other()
    average_win_1 = []
    average_win_2 = []
    for i in range(args.n_run):
        time_steps = 0
        current_episode = 1
        red_win_flag = 0
        blue_win_flag = 0
        draw_win_flag = 0
        episode_rewards = []
        while current_episode <= args.n_episodes:
            print("current_time_steps:{0}, current_episode:{1}".format(time_steps, current_episode))
            env.random_init()
            if current_episode % 50 == 0:
                env.reset(True)
            else:
                env.reset(False)
            termined = False
            agent_1.after_reset(env, "red")
            agent_2.after_reset(env, "blue")

            rewards = 0
            while not termined:

                agent_1.before_step_for_sample(env)
                agent_2.before_step_for_sample(env)

                env.step()
                time_steps += 1

                agent_1.after_step_for_sample(env)
                agent_2.after_step_for_sample(env)

                termined = env.done
                if termined:
                    win_flag = env.judge_red_win()
                    if win_flag == 1:
                        red_win_flag += 1
                    elif win_flag == 0:
                        draw_win_flag += 1
                    else:
                        blue_win_flag += 1
            current_episode += 1
        print('red_win_rate is ', red_win_flag / args.n_episodes, 'red_win_rate(draw) is ', (red_win_flag + draw_win_flag) / args.n_episodes)
        print('blue_win_rate is ', blue_win_flag / args.n_episodes, 'blue_win_rate(draw) is ', (blue_win_flag + draw_win_flag) / args.n_episodes)
        average_win_1.append(red_win_flag / args.n_episodes)
        average_win_2.append(blue_win_flag / args.n_episodes)
    print(sum(average_win_1) / len(average_win_1))
    print(sum(average_win_2) / len(average_win_2))

if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument("--save_gifs", action='store_true', default=False)
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--n_run", default=5, type=int)
    args = parser.parse_args()
    run(args)

