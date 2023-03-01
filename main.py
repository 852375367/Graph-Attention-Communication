
from config import Config
import numpy as np
from environment.battlespace import BattleSpace
from agent_baseline.Agent import Agent
from agent_baseline.simple_agent import Simple as Agent_other
import matplotlib.pyplot as plt

def main(config, env):
    red = "red"
    blue = "blue"
    evaluate1 = False
    evaluate2 = False
    agent_1 = Agent(config)
    agent_2 = Agent_other()
    time_steps = 0
    t_max = 2000000
    average_win = 0
    current_episode = 1
    T = []
    episode_rate = []
    while time_steps <= t_max:
        print("current_time_steps:{0}, current_episode:{1}, max_time_step:{2}".format(time_steps, current_episode, t_max))

        if current_episode % 100 == 0:
            average_win = agent_1.win_flag[-100:].count(1) / 100
            print("{0}-{1} episode agent_1 win rate : {2}".format(current_episode - 99, current_episode, average_win))
            print("{0}-{1} episode agent_2 win rate : {2}".format(current_episode - 99, current_episode, agent_1.win_flag[-100:].count(-1) / 100))
            print("{0}-{1} draw rate : {2}".format(current_episode - 99, current_episode, agent_1.win_flag[-100:].count(0) / 100))
            T.append(current_episode)
            episode_rate.append(average_win)

        env.random_init()
        env.reset(False)
        termined = False

        agent_1.after_reset(env, red)
        agent_2.after_reset(env, blue)

        while not termined:
            agent_1.before_step_for_sample(env)
            agent_2.before_step_for_sample(env)

            env.step()
            time_steps += 1
            if not evaluate1:
                agent_1.after_step_for_sample(env)
            # if not evaluate2:
            #     agent_2.after_step_for_sample(env)
            termined = env.done

        if not evaluate1:
            agent_1.before_step_for_train(env)
        if not evaluate2:
            agent_2.before_step_for_train(env)

        if not evaluate1 and current_episode % agent_1.args.buffer_size == 0:
            agent_1.replay_buffer.compute_returns()
            for train_step in range(agent_1.args.train_steps):
                mini_batch = agent_1.get_batchs()
                agent_1.train(mini_batch)
            agent_1.after_step_for_train(env)

        # if not evaluate2 and current_episode % agent_2.args.buffer_size == 0:
        #     agent_2.replay_buffer.compute_returns()
        #     for train_step in range(agent_2.args.train_steps):
        #         mini_batch = agent_2.get_batchs()
        #         agent_2.train(mini_batch)
        #     agent_2.after_step_for_train(env)
        current_episode += 1

if __name__ == '__main__':
    env = BattleSpace()
    cfg = Config()
    main(cfg, env)

