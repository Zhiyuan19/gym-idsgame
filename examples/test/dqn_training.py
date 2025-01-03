import sys
import os
from torch import nn
sys.path.insert(0, '/home/videoserver/.local/lib/python3.8/site-packages')
import numpy
#print("NumPy Version:", numpy.__version__)
import gymnasium as gym
import gym_idsgame
import numpy as np
from stable_baselines3 import DQN
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

policy_kwargs = dict(
    net_arch=[256, 256],  
    activation_fn=nn.ReLU  
)
#previous_run_id = "rheiyy13"
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "idsgame-random_attack-v22",
    "learning_rate": 1e-3,
    "buffer_size": 25000,
    "batch_size": 256,
    "gamma": 0.99,
    "train_freq": 256,  
    "gradient_steps": 128,  
    "exploration_fraction": 0.16,
    "exploration_final_eps": 0.04,
    "learning_starts": 1000, 
    "target_update_interval": 500
}
run = wandb.init(
    project="dqn_emulatednetwork_project",
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

    def _on_step(self) -> bool:
        if "ws_failattacks" in self.locals["infos"][-1] and self.locals["dones"][-1]:
            ws_failattacks = self.locals["infos"][-1]["ws_failattacks"]
            ws_successattacks = self.locals["infos"][-1]["ws_successattacks"]
            h3_failattacks = self.locals["infos"][-1]["h3_failattacks"]
            h3_successattacks = self.locals["infos"][-1]["h3_successattacks"]
            fw_failattacks = self.locals["infos"][-1]["fw_failattacks"]
            fw_successattacks = self.locals["infos"][-1]["fw_successattacks"]
            ma_failattacks = self.locals["infos"][-1]["ma_failattacks"]
            ma_successattacks = self.locals["infos"][-1]["ma_successattacks"]

            # recording
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

    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./dqnlogs/",
        log_path="./dqnlogs/",
        eval_freq=1000,
        callback_after_eval=stop_train_callback,
        deterministic=True,
        verbose=1,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=2500,
        save_path="./dqnlogs/",
        name_prefix="dqnrl_model",
        save_replay_buffer=True,
        #save_vecnormalize=True,
    )
    
    checkpoint_dir = "./dqnlogs/"
    checkpoint_prefix = "dqnrl_model"

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
        model = DQN.load(latest_checkpoint, env=env)
        remaining_timesteps = max(0, config["total_timesteps"] - model.num_timesteps)
        replay_buffer_file = latest_checkpoint.replace(".zip", "_replay_buffer.pkl")
        model.load_replay_buffer(replay_buffer_file)
    else:
        print("No checkpoint found. Starting new training.")
        remaining_timesteps = config["total_timesteps"]
        model = DQN(
            config["policy_type"], 
            env, 
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            train_freq=config["train_freq"],
            gradient_steps=config["gradient_steps"],
            exploration_fraction=config["exploration_fraction"],
            exploration_final_eps=config["exploration_final_eps"],
            learning_starts=config["learning_starts"],
            target_update_interval=config["target_update_interval"],
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            tensorboard_log=f"runs/{run.id}")
    
    print(f"Starting learn() from timesteps: {model.num_timesteps}")

    try:
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
        model.save("interrupted_model.zip")
    
    model.save(f"models/{run.id}_dqnfinal.zip")
    model.save_replay_buffer("models/dqn_replay_buffer")
    idsgame_env.is_end()
    run.finish()


