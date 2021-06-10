import os
import gym
import numpy as np
# Importing the PPO algorithm and the make_vec_env packages
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

from GeeseGymWrapper import HungryGeeseKaggle

import json

import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import ByteStorage, nn as nn
import torch.nn.functional as F

from torchsummary import summary


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
                    print(self.model.policy.state_dict().keys())
                    th.save(self.model.policy.state_dict(), f'tmp/best_pytorch_model')

        return True

class Net(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Net, self).__init__(observation_space, features_dim)
        self.conv1 = nn.Conv2d(7, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.fc3 = nn.Linear(2112, features_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc3(x))
        return x
    
# Create log dir
log_dir = "tmp/"


# We create multiple vectorized environments
#geese_env = make_vec_env(HungryGeeseKaggle, n_envs = 6)
#env = gym.make(HungryGeeseKaggle)
env = HungryGeeseKaggle()
env = Monitor(env, log_dir)

os.makedirs(log_dir, exist_ok=True)
# Create the callback: check every 1000 steps

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# We create the PPO agent and train it
'''
policy_kwargs = {
    'activation_fn':th.nn.ReLU, 
    'net_arch':[64, dict(pi=[32, 16], vf=[32, 16])],
    'features_extractor_class':Net,
}'''
model = PPO('MlpPolicy', env, verbose=1,
            tensorboard_log="ppo_geese_tensorboard/")
model.load('tmp/best_model.zip')
print(model)
print(model.policy)
print(model.get_parameters())
model.learn(total_timesteps=((1e6)*10000), callback=callback)
model.save("ppo_goose")
