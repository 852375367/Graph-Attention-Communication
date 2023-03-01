import torch

class AgentBase:
    def before_reset(self,env,side):
        raise NotImplementedError("Please Implement this method")

    def after_reset(self, env, side):
        raise NotImplementedError("Please Implement this method")

    def before_step_for_sample(self, env):
        raise NotImplementedError("Please Implement this method")

    def after_step_for_sample(self,env):
        raise NotImplementedError("Please Implement this method")

    def before_step_for_train(self, env):
        raise NotImplementedError("Please Implement this method")

    def get_batchs(self):
        raise NotImplementedError("Please Implement this method")

    def after_step_for_train(self, env):
        raise NotImplementedError("Please Implement this method")

    def train(self, batchs):
        raise NotImplementedError("Please Implement this method")

    def get_interval(self):
        raise NotImplementedError("Please Implement this method")

    def print_train_log(self):
        raise NotImplementedError("Please Implement this method")

