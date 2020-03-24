"""
A random atatck agent for the gym-idsgame environment
"""
from typing import Union
import numpy as np
from gym_idsgame.agents.bot_agent import BotAgent
from gym_idsgame.envs.dao.game_state import GameState
from gym_idsgame.envs.dao.game_config import GameConfig
from gym_idsgame.envs.util import idsgame_util

class RandomAttackBotAgent(BotAgent):
    """
    Class implementing a random attack policy: a policy where the attacker selects a random node out of its neighbors
    and a random attack type in each iteration
    """

    def __init__(self, game_config: GameConfig):
        """
        Constructor, initializes the policy

        :param game_config: the game configuration
        """
        super(RandomAttackBotAgent, self).__init__(game_config)

    def action(self, game_state: GameState) -> int:
        """
        Samples an action from the policy

        :param game_state: the game state
        :return: action_id
        """
        actions = list(range(self.game_config.num_actions))
        legal_actions = list(filter(lambda action: idsgame_util.is_attack_id_legal(action,
                                                                                   self.game_config,
                                                                                   game_state.attacker_pos), actions))
        action = np.random.choice(legal_actions)
        return action
