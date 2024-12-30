import sys
sys.path.insert(0, '/home/videoserver/.local/lib/python3.8/site-packages')
import numpy
import gymnasium as gym
import gym_idsgame
import numpy as np
from stable_baselines3 import PPO
from gym_idsgame.envs.variables import variable
from gym_idsgame.envs.constants import constants
import random

class SingleAgentEnvWrapper(gym.Env):

    def __init__(self, idsgame_env, attacker_action: int):
        self.attacker_action = attacker_action
        self.idsgame_env = idsgame_env
        self.observation_space = gym.spaces.Box(low=np.zeros((5, 5)), high=np.full((5, 5), 9), dtype=np.int32)
        self.action_space = idsgame_env.action_space

    def step(self, a: int):
        action = (self.attacker_action, a)
        #print("action pair:", action)
        obs, rewards, done, _, info = self.idsgame_env.step(action)
        #print("obs", obs)
        #print("obs[1]", obs[1])
        #obs = obs.astype(np.int32)
        return obs, rewards, _, done, info

    def reset(self, seed: int = 0):
        o, _ = self.idsgame_env.reset()
        #print("observation after reset:")
        #print(o[1])
        return o[1], {}

    def render(self, mode: str ='human'):
        self.idsgame_env.render()
    
    def is_end(self):
        return self.idsgame_env.is_end()


if __name__ == '__main__':
    idsgame_env = gym.make("idsgame-random_attack-v22")
    env = SingleAgentEnvWrapper(idsgame_env=idsgame_env, attacker_action=0)
    print("node information:")
    print("node number:", idsgame_env.idsgame_config.game_config.num_nodes )
    node_list = idsgame_env.idsgame_config.game_config.network_config.node_list
    print("node list:",node_list)
    obs = env.idsgame_env.get_observation()
    print("snort observation is", obs)
    env.reset()  
    idsgame_env.is_end()
