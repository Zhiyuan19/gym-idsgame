import sys
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
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 5000,
    "env_name": "idsgame-random_attack-v22",
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

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
    obs, _ = env.reset(seed=None)
    nodelist = [0, 1, 2, 3]
    typelist = [0, 1, 2, 3, 4]
    while env.idsgame_env.state.done == False:
        node = random.choice(nodelist)
        defend_type = random.choice(typelist)
        #action = [nodelist[env.idsgame_env.state.game_step], typelist[env.idsgame_env.state.game_step]]
        action = [node, defend_type]
        print("input action is", action)
        obs, reward, terminated, truncated, info = env.step(action)
        
    idsgame_env.is_end()

