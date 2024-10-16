import gymnasium as gym
import gym_idsgame
import numpy as np
from stable_baselines3 import PPO
from gym_idsgame.envs.variables import variable

class SingleAgentEnvWrapper(gym.Env):

    def __init__(self, idsgame_env, attacker_action: int):
        self.attacker_action = attacker_action
        self.idsgame_env = idsgame_env
        self.observation_space = gym.spaces.Box(low=np.zeros((5, 5)), high=np.full((5, 5), 9), dtype=np.int32)
        self.action_space = idsgame_env.action_space

    def step(self, a: int):
        action = (self.attacker_action, a)
        print("action pair:", action)
        obs, rewards, done, _, info = self.idsgame_env.step(action)
        #print("obs", obs)
        #print("obs[1]", obs[1])
        return obs[1], rewards[1], done, _, info

    def reset(self, seed: int = 0):
        o, _ = self.idsgame_env.reset()
        print("observation after reset:")
        print(o[1])
        return o[1], {}

    def render(self, mode: str ='human'):
        self.idsgame_env.render()


if __name__ == '__main__':
    idsgame_env = gym.make("idsgame-random_attack-v22")
    env = SingleAgentEnvWrapper(idsgame_env=idsgame_env, attacker_action=0)
    print("node information:")
    print("node number:", idsgame_env.idsgame_config.game_config.num_nodes )
    node_list = idsgame_env.idsgame_config.game_config.network_config.node_list
    print("node list:",node_list)
    print("attacker_pos:",idsgame_env.idsgame_config.game_config.network_config.start_pos)
    attacker_observation_space = idsgame_env.idsgame_config.game_config.get_attacker_observation_space()
    print("Attacker Observation Space:")
    print(attacker_observation_space)
    defender_observation_space = idsgame_env.idsgame_config.game_config.get_defender_observation_space()
    print("Defender Observation Space:")
    print(defender_observation_space)
    
    
    attacker_action_space = idsgame_env.idsgame_config.game_config.get_action_space(defender=False)
    print("Attacker Action Space:")
    print(attacker_action_space)
    defender_action_space = idsgame_env.idsgame_config.game_config.get_action_space(defender=True)
    print("Defender Action Space:")
    print(defender_action_space)
    
    print("Initial states information:")
    print("attack types number: ",idsgame_env.idsgame_config.game_config.num_attack_types)
    print("attack value matrix:")
    print(idsgame_env.idsgame_config.game_config.initial_state.attack_values)
    print("defense value matrix:")
    print(idsgame_env.idsgame_config.game_config.initial_state.defense_values)
    print("detection value matrix:")
    print(idsgame_env.idsgame_config.game_config.initial_state.defense_det)
    print("reconnaissance_state matrix:")
    print(idsgame_env.idsgame_config.game_config.initial_state.reconnaissance_state)
    variable_config = variable.VariableConfig()
    print("Priority after change:", variable_config.priority)
    obs, _ = env.reset()
    #obs, rewards, dones, _, info = env.step(4)
    #while True:
        #env.render("human")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    obs, _ = env.reset()
    #print("obs is", obs)
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, _, info = env.step(action)
        env.render("human")
