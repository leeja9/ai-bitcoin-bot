import numpy as np
import pandas as pd

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

import gym_anytrading
from gym_anytrading.envs import StocksEnv

import itertools



def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, feature_list].to_numpy()[start:end]
    return prices, signal_features

def calculate_log_returns(prices):
    return np.log(prices[1:] / prices[:-1])


# Copied directly from one of the last group's, for testing purposes only.
# We should make our own.
class BitcoinEnv(StocksEnv):
    _process_data = my_process_data

    def step(self, action):
        observation, reward, done, info = super().step(action)
        # Compute the Sharpe Ratio over the last 18 steps and use it as the reward
        if len(self.prices) >= 18:
            returns = np.diff(self.prices[-18:])
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            reward = sharpe_ratio

        return observation, reward, done, info

    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        sharpe_ratio = (mean_returns - risk_free_rate) / std_returns
        return sharpe_ratio

df = pd.read_csv("./data/indicators.csv")
df.dropna(inplace=True)

print(df.shape)


env = BitcoinEnv(df=df, window_size=28, frame_bound=(28, len(df)))
# Make this a vextorized env?


model = RecurrentPPO("MlpLstmPolicy")