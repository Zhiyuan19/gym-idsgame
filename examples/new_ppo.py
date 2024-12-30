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
    idsgame_env = gym.make("idsgame-random_attack-v22")
    #print("observation space is ", idsgame_env.observation_space)
    #print("action space is ", idsgame_env.action_space)
    env = SingleAgentEnvWrapper(idsgame_env=idsgame_env)
    #print("node information:")
    #print("node number:", idsgame_env.idsgame_config.game_config.num_nodes )
    #node_list = idsgame_env.idsgame_config.game_config.network_config.node_list
    #print("node list:",node_list)
    #print("attacker_pos:",idsgame_env.idsgame_config.game_config.network_config.start_pos)
    #attacker_observation_space = idsgame_env.idsgame_config.game_config.get_attacker_observation_space()
    #print("Attacker Observation Space:")
    #print(attacker_observation_space)
    #defender_observation_space = idsgame_env.idsgame_config.game_config.get_defender_observation_space()
    #print("Defender Observation Space:")
    #print(defender_observation_space)
    #print("env observation space is:")
    #print(env.observation_space)
    #print("env action space is:")
    #print(env.action_space)
    #attacker_action_space = idsgame_env.idsgame_config.game_config.get_action_space(defender=False)
    #print("Attacker Action Space:")
    #print(attacker_action_space)
    #defender_action_space = idsgame_env.idsgame_config.game_config.get_action_space(defender=True)
    #print("Defender Action Space:")
    #print(defender_action_space)
    
    #print("Initial states information:")
    #print("attack types number: ",idsgame_env.idsgame_config.game_config.num_attack_types)
    #print("attack value matrix:")
    #print(idsgame_env.idsgame_config.game_config.initial_state.attack_values)
    #print("defense value matrix:")
    #print(idsgame_env.idsgame_config.game_config.initial_state.defense_values)
    #print("detection value matrix:")
    #print(idsgame_env.idsgame_config.game_config.initial_state.defense_det)
    #print("reconnaissance_state matrix:")
    #print(idsgame_env.idsgame_config.game_config.initial_state.reconnaissance_state)
    #print("apt phase for all the nodes:", idsgame_env.state.apt_stage)
    #attack_id = idsgame_env.idsgame_config.attacker_agent.action(idsgame_env.state)
    #print("attack_id is", attack_id)
    #variable_config = variable.VariableConfig()
    #print("Priority after change:", variable_config.priority)
    nodelist = [0, 1, 2, 3]
    typelist = [0, 1, 2, 3, 4]

    obs, _ = env.reset(seed=None)
    
    while env.idsgame_env.state.done == False:
        node = random.choice(nodelist)
        defend_type = random.choice(typelist)
        #action = [nodelist[env.idsgame_env.state.game_step], typelist[env.idsgame_env.state.game_step]]
        action = [node, defend_type]
        print("input action is", action)
        obs, reward, terminated, truncated, info = env.step(action)
        
    obs, _ = env.reset()
    while env.idsgame_env.state.done == False:
        node = random.choice(nodelist)
        defend_type = random.choice(typelist)
        #action = [nodelist[env.idsgame_env.state.game_step], typelist[env.idsgame_env.state.game_step]]
        action = [node, defend_type]
        print("input action is", action)
        obs, reward, terminated, truncated, info = env.step(action)
    
    #while env.idsgame_env.state.game_step <= 8:
        #action = [nodelist[env.idsgame_env.state.game_step], typelist[env.idsgame_env.state.game_step]]
        #print("input action is", action)
        #obs, rewards, dones, info = env.step(action)
    
    #obs, _ = env.reset()
    
    #while env.idsgame_env.state.game_step <= 8:
        #action = [nodelist[env.idsgame_env.state.game_step], typelist[env.idsgame_env.state.game_step]]
        #print("input action is", action)
        #obs, rewards, dones, info = env.step(action)
        
    idsgame_env.is_end()
    #obs, rewards, dones, _, info = env.step(4)
    #while True:
        #env.render("human")

    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=1000)
    #model.save("ppo_single_agent_env")
    #model = PPO.load("ppo_single_agent_env", env=env)
    #obs, _ = env.reset()
    ##print("obs is", obs)
    #while True:
        #action, _states = model.predict(obs)
        #obs, rewards, dones, _, info = env.step(action)
        #env.render("human")
