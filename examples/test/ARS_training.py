import sys
import os
from torch import nn
sys.path.insert(0, '/home/videoserver/.local/lib/python3.8/site-packages')
import numpy
#print("NumPy Version:", numpy.__version__)
import gymnasium as gym
import gym_idsgame
import numpy as np
from sb3_contrib import ARS
from gym_idsgame.envs.variables import variable
from gym_idsgame.envs.constants import constants
import random
from typing import Union
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.vec_env import VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from collections import deque

policy_kwargs = dict(
    net_arch=[64, 64],  
    activation_fn=nn.ReLU  
)
#previous_run_id = "icuqzcys"
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "idsgame-random_attack-v22",
    "normalize": "dict(norm_obs=True, norm_reward=False)",
    "learning_rate": 0.02,
    "n_delta": 8,
    "n_top": 4,
    "delta_std": 0.025,
    "zero_policy": False,  
    "alive_bonus_offset": -0.1,
}
run = wandb.init(
    project="ars_emulatednetwork_project",
    #id=previous_run_id,  
    #resume="allow",      
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)
        
class EpisodeEndLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeEndLoggingCallback, self).__init__(verbose)
        self.success_counter = deque(maxlen=100)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        #print("Type of env:", type(env))
        current_env = env
        while hasattr(current_env, "env"):
            current_env = current_env.env
            #print("Wrapped by:", type(current_env))
            
        #print("failed number is", current_env.idsgame_env.unwrapped.num_fail_attacks)
        #print("success number is", current_env.idsgame_env.unwrapped.num_success_attacks)
        #print("episode number is", current_env.idsgame_env.unwrapped.episodecounter)
        #print("done is",current_env.idsgame_env.unwrapped.episodedone)
        if current_env.idsgame_env.unwrapped.episodedone == True:
            ws_failattacks = current_env.idsgame_env.unwrapped.num_fail_attacks[0]
            ws_successattacks = current_env.idsgame_env.unwrapped.num_success_attacks[0]
            h3_failattacks = current_env.idsgame_env.unwrapped.num_fail_attacks[1]
            h3_successattacks = current_env.idsgame_env.unwrapped.num_success_attacks[1]
            fw_failattacks = current_env.idsgame_env.unwrapped.num_fail_attacks[2]
            fw_successattacks = current_env.idsgame_env.unwrapped.num_success_attacks[2]
            ma_failattacks = current_env.idsgame_env.unwrapped.num_fail_attacks[3]
            ma_successattacks = current_env.idsgame_env.unwrapped.num_success_attacks[3]
            if current_env.idsgame_env.unwrapped.successflag == True:
                self.success_counter.append(1)
            else:
                self.success_counter.append(0)
                         
            if current_env.idsgame_env.unwrapped.episodecounter >=10:
                success_rate = sum(self.success_counter) / len(self.success_counter)
            else:
                success_rate = 0
            # recording
            self.logger.record("custom/success_rate", success_rate)
            self.logger.record("custom/ws_failattacks", ws_failattacks)
            self.logger.record("custom/ws_successattacks", ws_successattacks)
            self.logger.record("custom/h3_failattacks", h3_failattacks)
            self.logger.record("custom/h3_successattacks", h3_successattacks)
            self.logger.record("custom/fw_failattacks", fw_failattacks)
            self.logger.record("custom/fw_successattacks", fw_successattacks)
            self.logger.record("custom/ma_failattacks", ma_failattacks)
            self.logger.record("custom/ma_successattacks", ma_successattacks)

            if self.verbose > 0:
                print(f"Episode End | "
                      f"WS Fail: {ws_failattacks}, WS Success: {ws_successattacks}, "
                      f"H3 Fail: {h3_failattacks}, H3 Success: {h3_successattacks}, "
                      f"FW Fail: {fw_failattacks}, FW Success: {fw_successattacks}, "
                      f"MA Fail: {ma_failattacks}, MA Success: {ma_successattacks}")

        return True
    
class SingleAgentEnvWrapper(gym.Env):

    def __init__(self, idsgame_env):
        #self.attacker_action = attacker_action
        self.idsgame_env = idsgame_env
        #self.observation_space = gym.spaces.Box(low=np.zeros((5, 5)), high=np.full((5, 5), 9), dtype=np.int32)
        self.observation_space = idsgame_env.observation_space
        self.num_hosts = idsgame_env.action_space.nvec[0]  # self.number_hosts - 3 = 4
        self.num_defend_actions = idsgame_env.action_space.nvec[1]  # self.number_defendtype + 1 = 5
        self.action_space = gym.spaces.Discrete(self.num_hosts * self.num_defend_actions)
        #self.num_fail_attacks = idsgame_env.unwrapped.num_fail_attacks
        #self.num_success_attacks = idsgame_env.unwrapped.num_success_attacks
        #self.done = idsgame_env.unwrapped.state.done
        #self.episodecounter = idsgame_env.unwrapped.episodecounter

    def step(self, action: int):
        host_id = action // self.num_defend_actions
        defend_action_id = action % self.num_defend_actions
        decoded_action = [host_id, defend_action_id]
        #print("action pair:", action)
        obs, reward, terminated, truncated, info = self.idsgame_env.step(decoded_action)
        #print("obs", obs)
        #print("obs[1]", obs[1])
        #obs = obs.astype(np.int32)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
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
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    eval_env = env

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./ars_{run.id}_secondlogs/",
        log_path=f"./ars_{run.id}_secondlogs/",
        eval_freq=1000,
        callback_after_eval=stop_train_callback,
        deterministic=True,
        verbose=1,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=2500,
        save_path=f"./ars_{run.id}_secondlogs/",
        name_prefix=f"ars_{run.id}_secondrl_model",
        #save_replay_buffer=True,
        #save_vecnormalize=True,
    )
    
    checkpoint_dir = f"./ars_{run.id}_secondlogs/"
    checkpoint_prefix = f"ars_{run.id}_secondrl_model"

    # finding checkpoints
    def find_latest_checkpoint(checkpoint_dir, prefix):
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
            return None
        files = os.listdir(checkpoint_dir)
        checkpoint_files = [f for f in files if f.startswith(prefix) and f.endswith(".zip")]
        if not checkpoint_files:
            return None
        checkpoint_files.sort(key=lambda f: int(f.split('_')[-2]))
        return os.path.join(checkpoint_dir, checkpoint_files[-1])

    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, checkpoint_prefix)
    
    if latest_checkpoint:
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        model = ARS.load(latest_checkpoint, env=env)
        remaining_timesteps = max(0, config["total_timesteps"] - model.num_timesteps)
    else:
        print("No checkpoint found. Starting new training.")
        remaining_timesteps = config["total_timesteps"]
        model = ARS(
            config["policy_type"], 
            env, 
            learning_rate=config["learning_rate"],
            n_delta=config["n_delta"],
            n_top=config["n_top"],
            delta_std=config["delta_std"],
            zero_policy=config["zero_policy"],
            alive_bonus_offset=config["alive_bonus_offset"],
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            tensorboard_log=f"runs/{run.id}")
    
    print(f"Starting learn() from timesteps: {model.num_timesteps}")

    try:
        print("remaining times", remaining_timesteps)
        model.learn(
            total_timesteps=remaining_timesteps, 
            progress_bar=True, 
            reset_num_timesteps= False,
            callback=[WandbCallback(
                gradient_save_freq=5,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ), eval_callback, checkpoint_callback, EpisodeEndLoggingCallback(verbose=0)]
        )
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving model...")
        model.save("interrupted_ars_model.zip")
    
    model.save(f"models/{run.id}_arsfinal.zip")
    idsgame_env.unwrapped.is_end()
    run.finish()


