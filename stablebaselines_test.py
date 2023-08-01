import gym

from stable_baselines.common_policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2


from gym_trading_env.utils.history import History
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym
import gym_trading_env
from gym_trading_env.utils.history import History
from gym_trading_env.downloader import download
import datetime


# load data
df = pd.read_csv("./data/indicators.csv")
df.dropna(inplace=True)


def add_reward_columns(df: pd.DataFrame):
    """add reward columns to dataframe for incremental updates"""
    for col in ['lr', 'alr', 'var_sum']:
        df[col] = 0


def update_reward_columns(history: History) -> None:
    """Set this episode lr, alr, var_sum, sr, powc"""

    # Using weighted incremental algorithmic approach for average
    # https://math.stackexchange.com/questions/106700/incremental-averaging
    # general formula is: mean = ((n - 1) * last_mean + this_value) / n))

    # logarithmic return
    this_lr = 0
    # if position is 1 (100% BTC)
    if history['position', -1] == 1:
        this_lr = np.log(history['data_close', -1]) - \
            np.log(history['data_close', -2])
    history.__setitem__(('data_lr', -1), this_lr)  # update history with new lr

    # running average of logarithmic return
    n = len(history)
    last_alr = history['data_alr', -2]
    this_alr = ((n - 1) * last_alr + this_lr) / n
    # update history with new alr
    history.__setitem__(('data_alr', -1), this_alr)

    # running variance sum of logarithmic return
    # for each nth row, dividing this sum by n gives population variance
    last_alr = history['data_alr', -2]
    last_var_sum = history['data_var_sum', -2]
    this_var_sum = last_var_sum + \
        abs((this_lr - last_alr) * (this_lr - this_alr))
    history.__setitem__(('data_var_sum', -1), this_var_sum)


def get_random_weights(arr_len):
    """get numpy array of random weights"""
    max_val = 100
    weight_vector = np.zeros(arr_len)
    for i in range(arr_len - 1):
        n = np.random.randint(0, max_val)
        max_val = max_val - n
        weight_vector[i] = n
    weight_vector /= 100
    weight_vector[-1] = 1 - sum(weight_vector[:-1])
    np.random.shuffle(weight_vector)
    return weight_vector


def reward_function(history: History) -> float:
    """reward function for gym-trading-env"""
    update_reward_columns(history)
    average_log_return = history['data_alr', -1]
    var_sum = history['data_var_sum', -1]
    variance = var_sum / len(history)
    std_dev = np.sqrt(variance)
    sharpe_ratio = average_log_return / 0.5
    this_lr = history['data_lr', -1]
    powc = 0
    # if this eposide position is 0 (100% USD) and last position was 1 (100% BTC)
    # this compute time can also be traded for memory by adding a tracking column if needed
    if (history['position', -1] == 0 and history['position', -2] == 1):
        idx = history[-2]['idx']

        # This is an infinite loop if idx == 0 and history['position', idx] != 0.
        while idx >= 0:
            if (history['position', idx] == 0):
                last_lr = history['data_lr', idx + 1]
                powc = this_lr - last_lr
    reward_vector = np.array([average_log_return, sharpe_ratio, powc])
    weight_vector = get_random_weights(len(reward_vector))
    # dot product of random weights and reward values
    reward = reward_vector @ weight_vector
    return reward


def dynamic_features(history: History) -> float:
    """Calculates dynamic features."""
    # dyn_features = [last_position, real_position]
    # return dyn_features

    pass


env = env = gym.make("TradingEnv",
                     name="BTCUSD",
                     df=df,  # Your dataset with your custom features
                     # -1 (=SHORT), 0(=SELL ALL), +1 (=BUY ALL)
                     positions=[0, 1],
                     # trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
                     # borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
                     # dynamic_feature_functions = [dynamic_features]
                     reward_function=reward_function,
                     portfolio_initial_value=10000,
                     # max_episode_duration = 1000,
                     )

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000) 
model.save("ppo2_trading_env")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()