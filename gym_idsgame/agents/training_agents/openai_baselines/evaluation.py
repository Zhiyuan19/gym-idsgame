# Copied from stable_baselines
import numpy as np
import torch as th
import time
from stable_baselines3.common.vec_env import VecEnv
from gym_idsgame.agents.training_agents.policy_gradient.pg_agent_config import PolicyGradientAgentConfig
from gym_idsgame.envs.constants import constants

def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
                    render=False, callback=None, reward_threshold=None,
                    return_episode_rewards=False, pg_agent_config : PolicyGradientAgentConfig = None,
                    train_episode = 1):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    pg_agent_config.logger.info("Starting Evaluation")
    time_str = str(time.time())

    model.num_eval_games = 0
    model.num_eval_hacks = 0

    # if len(self.eval_result.avg_episode_steps) > 0:
    #     self.config.logger.warning("starting eval with non-empty result object")
    if pg_agent_config.eval_episodes < 1:
        return
    done = False

    # Tracking metrics
    episode_attacker_rewards = []
    episode_defender_rewards = []
    episode_steps = []

    env.envs[0].enabled = True
    env.envs[0].stats_recorder.closed = False
    env.envs[0].episode_id = 0

    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        while not done:
            if pg_agent_config.eval_render:
                env.render()
                time.sleep(pg_agent_config.eval_sleep)

            # Get attacker and defender actions
            obs = th.tensor(obs)
            res = model.predict(obs, deterministic=False)
            action = res.numpy()

            # Take a step in the environment
            obs, reward, done, _info = env.step(np.array([action]))

            # Update state information and metrics
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
        # Render final frame when game completed
        if pg_agent_config.eval_render:
            env.render()
            time.sleep(pg_agent_config.eval_sleep)
        pg_agent_config.logger.info("Eval episode: {}, Game ended after {} steps".format(episode, episode_length))

        # Record episode metrics
        episode_attacker_rewards.append(episode_reward)
        episode_steps.append(episode_length)

        # Update eval stats
        model.num_eval_games += 1
        model.num_eval_games_total += 1
        if env.envs[0].idsgame_env.state.detected:
            model.eval_attacker_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
            model.eval_defender_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
        if env.envs[0].idsgame_env.state.hacked:
            model.eval_attacker_cumulative_reward += constants.GAME_CONFIG.POSITIVE_REWARD
            model.eval_defender_cumulative_reward -= constants.GAME_CONFIG.POSITIVE_REWARD
            model.num_eval_hacks += 1
            model.num_eval_hacks_total += 1

        # Log average metrics every <self.config.eval_log_frequency> episodes
        if episode % pg_agent_config.eval_log_frequency == 0:
            if model.num_eval_hacks > 0:
                model.eval_hack_probability = float(model.num_eval_hacks) / float(model.num_eval_games)
            if model.num_eval_games_total > 0:
                model.eval_cumulative_hack_probability = float(model.num_eval_hacks_total) / float(
                    model.num_eval_games_total)
            model.log_metrics(train_episode, model.eval_result, episode_attacker_rewards, episode_defender_rewards,
                             episode_steps, eval=True, update_stats=False)

        # Save gifs
        if pg_agent_config.gifs and pg_agent_config.video:
            # Add frames to tensorboard
            for idx, frame in enumerate(env.envs[0].episode_frames):
                model.tensorboard_writer.add_image(str(train_episode) + "_eval_frames/" + str(idx),
                                                   frame, global_step=train_episode,
                                                   dataformats="HWC")

            # Save Gif
            env.envs[0].generate_gif(pg_agent_config.gif_dir + "episode_" + str(train_episode) + "_"
                                  + time_str + ".gif", pg_agent_config.video_fps)

        # Reset for new eval episode
        done = False
        obs = env.reset()

    # Log average eval statistics
    if model.num_eval_hacks > 0:
        model.eval_hack_probability = float(model.num_eval_hacks) / float(model.num_eval_games)
    if model.num_eval_games_total > 0:
        model.eval_cumulative_hack_probability = float(model.num_eval_hacks_total) / float(
            model.num_eval_games_total)

    model.log_metrics(train_episode, model.eval_result, episode_attacker_rewards, episode_defender_rewards,
                     episode_steps, eval=True, update_stats=True)


    mean_reward = np.mean(episode_attacker_rewards)
    std_reward = np.std(episode_attacker_rewards)

    pg_agent_config.logger.info("Evaluation Complete")
    print("Evaluation Complete")
    env.close()
    return mean_reward, std_reward