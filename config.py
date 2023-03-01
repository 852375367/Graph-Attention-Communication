import argparse

class Config:
    def __init__(self):
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

        parser.add_argument("--n_attention", default=64, type=int)
        parser.add_argument("--gumbel_softmax_tau", default=0.01, type=float)

        self.args = parser.parse_args()
