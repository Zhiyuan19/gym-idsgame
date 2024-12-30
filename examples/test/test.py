import sys
import os
from torch import nn
sys.path.insert(0, '/home/videoserver/.local/lib/python3.8/site-packages')
import numpy
#print("NumPy Version:", numpy.__version__)
import gymnasium as gym
import gym_idsgame
import numpy as np
from stable_baselines3 import PPO
from gym_idsgame.envs.variables import variable
from gym_idsgame.envs.constants import constants
import random
from typing import Union
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "idsgame-random_attack-v22",
}

class SingleAgentEnvWrapper(gym.Env):

    def __init__(self, idsgame_env):
        #self.attacker_action = attacker_action
        self.idsgame_env = idsgame_env
        #self.observation_space = gym.spaces.Box(low=np.zeros((5, 5)), high=np.full((5, 5), 9), dtype=np.int32)
        self.observation_space = idsgame_env.observation_space
        self.action_space = idsgame_env.action_space

    def step(self, a: Union[np.ndarray, list]):
        action = a
        #print("action pair:", action)
        obs, reward, terminated, truncated, info = self.idsgame_env.step(action)
        #print("obs", obs)
        #print("obs[1]", obs[1])
        #obs = obs.astype(np.int32)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        o, _ = self.idsgame_env.reset(seed=seed, options=options)
        return o, {}

    def render(self, mode: str ='human'):
        self.idsgame_env.render()
    
    def is_end(self):
        return self.idsgame_env.is_end()
if __name__ == '__main__':
    idsgame_env = gym.make(config["env_name"])
    env = SingleAgentEnvWrapper(idsgame_env=idsgame_env)
    env = Monitor(env, info_keywords=("is_success", "ws_failattacks", "ws_successattacks",
    "h3_failattacks", "h3_successattacks",
    "fw_failattacks", "fw_successattacks",
    "ma_failattacks", "ma_successattacks",))  # record stats such as returns
    
    eval_env = env
    checkpoint_dir = "./ppologs/"
    checkpoint_prefix = "pporl_model"
    # finding checkpoints
    def find_latest_checkpoint(checkpoint_dir, prefix):
        files = os.listdir(checkpoint_dir)
        checkpoint_files = [f for f in files if f.startswith(prefix) and f.endswith(".zip")]
        if not checkpoint_files:
            return None
        checkpoint_files.sort(key=lambda f: int(f.split('_')[-2]))
        return os.path.join(checkpoint_dir, checkpoint_files[-1])

    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, checkpoint_prefix)
    
    if latest_checkpoint:
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
    else:
        print("No checkpoint found. Starting new training.")
        model = PPO(config["policy_type"], env, learning_rate=1e-4, batch_size=250, n_steps=500, ent_coef=1e-4, clip_range=0.2, gae_lambda=0.95, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"runs/{run.id}")
        
    print(f"Loaded model with timesteps: {model.num_timesteps}")

