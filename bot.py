# Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
import stable_baselines3
import gym_trading_env
from gym_trading_env.renderer import Renderer   
from pathlib import Path
from BitcoinDownloader import download_exchange_data, get_dataframes
from BitcoinIndicators import Indicators
from BitcoinRewards import reward_function

import torch
torch.device("cuda" if torch.cuda.is_available() else "cpu")
get_ipython().run_line_magic('matplotlib', 'inline')


# Download Data if needed

download_again = False
data_path = './data/binance-BTCUSDT-1h.pkl'
if not Path(data_path).is_file() or download_again:
    download_exchange_data()
else:
    print("data already downloaded")


training_df, testing_df = get_dataframes()
print(training_df.head(2))
print(training_df.tail(2))
print(testing_df.head(2))
print(testing_df.tail(2))


# Create features

ind_path = './data/indicators.csv'
training_df.to_csv(ind_path)
indicators = Indicators(ind_path)
indicators.to_csv(ind_path)
training_df = pd.read_csv(ind_path)
training_df["date_open"] = pd.to_datetime(training_df["date_open"])
training_df.set_index("date_open", inplace=True)

training_df.dropna(inplace=True)
print(training_df.head(3))
print(training_df.tail(3))


# Create Environment

training_env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = training_df, # Your dataset with your custom features
        positions = [0, 1], # -1 (=SHORT), 0(=SELL ALL), +1 (=BUY ALL)
        #trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        #borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
        #dynamic_feature_functions = [dynamic_features]
        # reward_function = reward_function,
        portfolio_initial_value = 1000,
        reward_function = reward_function
        #max_episode_duration = 1000,
    )

observation, info = training_env.reset()
print(observation)


# Create Model

model = RecurrentPPO('MlpLstmPolicy', # feed-forward neural network with multiple hidden layers
            training_env, # environment in which the agent interacts and learns
            verbose=1, # enables the training progress to be printed during the learning process
            gamma=0.95, # determines the importance of future rewards compared to immediate rewards
            n_steps=15, # steps to collect samples from the environment before performing an update
            ent_coef=0.01, # encourages exploration by adding entropy to the policy loss
            learning_rate=0.001, # controls the step size at which model's parameters are updated based on the gradient of the loss function
            clip_range=0.1, # limits the update to a certain range to prevent large policy updates
            batch_size=15,
            device='cuda' if torch.cuda.is_available() else 'cpu'
n = len(training_df)
model.learn(total_timesteps=n)


# Calculate indicators for testing dataset

ind_path = './data/indicators.csv'
testing_df.to_csv(ind_path)
indicators = Indicators(ind_path)
indicators.to_csv(ind_path)
testing_df = pd.read_csv(ind_path)
testing_df["date_open"] = pd.to_datetime(testing_df["date_open"])
testing_df.set_index("date_open", inplace=True)

testing_df.dropna(inplace=True)
print(testing_df.head(3))
print(testing_df.tail(3))


testing_env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = testing_df, # Your dataset with your custom features
        positions = [0, 1], # -1 (=SHORT), 0(=SELL ALL), +1 (=BUY ALL)
        portfolio_initial_value = 1000,
    )

observation, info = testing_env.reset()
print(observation.shape)
print(observation)
print(info)


# Test trained model on testing data

for _ in range(len(testing_df)):
    position_index, _states = model.predict(observation)
    observation, reward, done, truncated, info = testing_env.step(position_index)
    testing_env.save_for_render(dir = "render_logs")
    if done or truncated:
        break


# Render results

renderer = Renderer(render_logs_dir="render_logs")
renderer.run()

