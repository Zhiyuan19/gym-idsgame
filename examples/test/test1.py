import gymnasium as gym
from stable_baselines3 import PPO
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
    "env_name": "CartPole-v1",
}
run = wandb.init(
    project="sb3",
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

            # 记录到日志系统
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
        
class SuccessRateWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SuccessRateWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["is_success"] = reward > 195  
        return obs, reward, terminated, truncated, info


env = SuccessRateWrapper(gym.make(config["env_name"]))
env = Monitor(env, info_keywords=("is_success", "ws_failattacks", "ws_successattacks",
    "h3_failattacks", "h3_successattacks",
    "fw_failattacks", "fw_successattacks",
    "ma_failattacks", "ma_successattacks",))  # record stats such as returns

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
    ), eval_callback, checkpoint_callback, EpisodeEndLoggingCallback(verbose=0)]
)
run.finish()
