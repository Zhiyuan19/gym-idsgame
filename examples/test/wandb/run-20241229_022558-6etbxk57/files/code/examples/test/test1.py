import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 5000,
    "env_name": "CartPole-v1",
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

class SuccessRateWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SuccessRateWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["is_success"] = reward > 195  
        return obs, reward, terminated, truncated, info


env = SuccessRateWrapper(gym.make(config["env_name"]))
env = Monitor(env, info_keywords=("is_success",))  # record stats such as returns

#eval_env = SuccessRateWrapper(gym.make(config["env_name"]))
eval_env = env

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=1000,
    callback_after_eval=stop_train_callback,
    deterministic=True,
    verbose=1,
    render=False,
)

checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logs/",
  name_prefix="rl_model",
  #save_replay_buffer=True,
  #save_vecnormalize=True,
)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"], 
    progress_bar=True, 
    callback=[WandbCallback(
        gradient_save_freq=1,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ), eval_callback, checkpoint_callback]
)
run.finish()
