"""
A random atatck agent for the gym-idsgame environment
"""
import numpy as np
from gym_idsgame.agents.bot_agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.util import idsgame_util
import random


class RandomAttackBotAgent(BotAgent):
    """
    Class implementing a random attack policy: a policy where the attacker selects a random node out of its neighbors
    and a random attack type in each iteration
    """

    def __init__(self, game_config: GameConfig, env):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(RandomAttackBotAgent, self).__init__(game_config)
        self.idsgame_env = env
        self.startattack = False
        
        
    def select(self, game_state: GameState):
        if self.startattack == False:
            flag = random.choices([0, 1], weights=[2, 3], k=1)[0]
            print(" start flag is", flag)
            if flag == 0: ## keep idle
                return [0, 5]
            else: #start attacks
                self.startattack = True
                nodes = [0, 1, 2, 3]
                weights = [0.05, 0.05, 0.4, 0.5]
                attacknode = random.choices(nodes, weights=weights, k=1)[0]
                return [attacknode, 4] #start reconnaissance first  
        else:
            if game_state.attackresult == True:
                if game_state.apt_stage[game_state.attacknode] == 4: #already reach a impact state
                    possible_nodes = [node for node in [0, 1, 2, 3] if node != game_state.attacknode]
                    if possible_nodes[0] == 0:
                        return [0, 4]
                    else:
                        attacknode = random.choice(possible_nodes)  # Randomly choose another node
                        if game_state.apt_stage[attacknode] == 4:
                            return [attacknode, 3]
                        else:
                            return [attacknode, 4] # Start reconnaissance on a new node
                elif game_state.apt_stage[game_state.attacknode] == 3: #in root state
                    attacktype = random.choices([2, 3], weights=[2, 5], k=1)[0] # choose persistence or impact attack
                    return [game_state.attacknode, attacktype]
                elif game_state.apt_stage[game_state.attacknode] == 2: # in compromised state
                    attacktype = 1 # choose root attack
                    return [game_state.attacknode, attacktype]
                else: #in known state
                    if game_state.specialservice == True:
                        attacktype = random.choices([0, 3], weights=[2, 3], k=1)[0]
                    else:
                        attacktype = 0 #bruteforce attack
                    return [game_state.attacknode, attacktype]
            else: # attack failed
                nodes = [0, 1, 2, 3]
                weights = [0.15, 0.05, 0.4, 0.4]
                attacknode = random.choices(nodes, weights=weights, k=1)[0]
                #attacknode = random.choice([0, 1, 2, 3])
                if game_state.apt_stage[attacknode] == 4:
                    return [attacknode, 3]
                else:
                    return [attacknode, 4]
            
        
    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy

        :param game_state: the game state
        :return: action_id
        """
        from gym_idsgame.envs.util import idsgame_util
        actions = list(range(self.game_config.num_attack_actions))
        servernodes = list(range(self.game_config.num_nodes-1))
        print("the list of the servernodes", servernodes)
        if not self.game_config.reconnaissance_actions:
            legal_actions = list(filter(lambda action: idsgame_util.is_attack_id_legal(action,
                                                                                       self.game_config,
                                                                                       game_state.attacker_pos,
                                                                                       game_state), actions))
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
            else:
                action = np.random.choice(actions)
        else:
            attacker_obs = game_state.get_attacker_observation(
                self.game_config.network_config, local_view=self.idsgame_env.local_view_features(),
                reconnaissance=self.game_config.reconnaissance_actions,
                reconnaissance_bool_features=self.idsgame_env.idsgame_config.reconnaissance_bool_features)
            #legal_actions = list(
                #filter(lambda action: self.is_attack_legal(action, attacker_obs, game_state), actions))
            legal_actions = actions
            if len(legal_actions) > 0:
                servernode = np.random.choice(servernodes)
                if game_state.apt_stage[servernode] == "reconnaissance":
                    action = servernode * (self.game_config.num_attack_types+1)+self.game_config.num_attack_types
                    print("the chosen node for attack is", servernode)
                    print("the chosen node type is", self.game_config.network_config.node_list[servernode])
                if game_state.apt_stage[servernode] == "initialaccess":
                    action = servernode * (self.game_config.num_attack_types+1)
                if game_state.apt_stage[servernode] == "privilegeescalation":
                    action = servernode * (self.game_config.num_attack_types+1)+1
                if game_state.apt_stage[servernode] == "persistence":
                    action = servernode * (self.game_config.num_attack_types+1)+2
                if game_state.apt_stage[servernode] == "excution":
                    action = servernode * (self.game_config.num_attack_types+1)+3
                #action = np.random.choice(legal_actions)
            else:
                servernode = np.random.choice(servernodes) #randomly chose a servernode to attack for attacker
                #print("the chosen node for attack is", servernode)
                #action = np.random.choice(actions)
            if self.idsgame_env.local_view_features():
                action = self.convert_local_attacker_action_to_global(action, attacker_obs)
        return action

    def is_attack_legal(self, action, obs, game_state):
        if self.idsgame_env.local_view_features():
            action = self.convert_local_attacker_action_to_global(action, obs)
            if action == -1:
                return False
        return idsgame_util.is_attack_id_legal(action, self.game_config,
                                       game_state.attacker_pos, game_state, [])

    def convert_local_attacker_action_to_global(self, action_id, attacker_obs):
        num_attack_types = self.idsgame_env.idsgame_config.game_config.num_attack_types
        neighbor = action_id // (num_attack_types + 1)
        attack_type = action_id % (num_attack_types + 1)
        target_id = int(attacker_obs[neighbor][num_attack_types])
        if target_id == -1:
            return -1
        attacker_action = target_id * (num_attack_types + 1) + attack_type
        return attacker_action
