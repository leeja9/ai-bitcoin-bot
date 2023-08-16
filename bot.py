import os
from datetime import datetime
import pandas as pd
import gymnasium as gym
from sb3_contrib import RecurrentPPO
import stable_baselines3 as sb3
import gym_trading_env
import torch
from pathlib import Path
from BitcoinDownloader import download_exchange_data, get_dataframes
from BitcoinIndicators import Indicators
from BitcoinRewards import reward_function
from BitcoinMetrics import add_env_metrics
from BitcoinRenderer import render_saved_logs

print("# Starting bot...")

# Globals
# These can be refactored into CLI arguments with argparse
TRAIN_NEW_MODEL = False
DOWNLOAD_NEW_DATA = False
RENDER_RESULTS = True
N_TIMESTEPS = 100000  # number of training timesteps
SAVED_MODEL = 'RecurrentPPO-2023-08-14T02-16-16'
MODEL_PATH = f'./saved_models/{SAVED_MODEL}'
RENDER_DIR = 'render_logs'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DOWNLOAD_NEW_DATA:
    print("# Downloading data...")
    data_dir = './data/'
    data_name = 'binance-BTCUSDT-1h.pkl'
    data_path = data_dir + data_name
    os.makedirs(data_dir, exist_ok=True)
    if not Path(data_path).is_file():
        download_exchange_data()
    else:
        print("data already downloaded")

print("# Split data into training and test dataframes...")
training_df, testing_df = get_dataframes()

if TRAIN_NEW_MODEL:
    print("# Calculate state vector for training dataframe...")
    ind_path = './data/indicators.csv'
    training_df.to_csv(ind_path)
    indicators = Indicators(ind_path)
    indicators.to_csv(ind_path)
    training_df = pd.read_csv(ind_path)
    training_df["date_open"] = pd.to_datetime(training_df["date_open"])
    training_df.set_index("date_open", inplace=True)
    training_df.dropna(inplace=True)
    print("Done")

    print("# Creating training environment...")
    training_env = gym.make("TradingEnv",
                            name="BTCUSD",
                            df=training_df,
                            positions=[0, 1],  # 0(=SELL ALL), +1 (=BUY ALL)
                            initial_position=0,
                            portfolio_initial_value=1000,
                            reward_function=reward_function)
    observation, info = training_env.reset()
    print("Done")

    print("# Create RL Model...")
    model = RecurrentPPO('MlpLstmPolicy',  # feed-forward neural network with multiple hidden layers
                         env=training_env,  # environment in which the agent interacts and learns
                         verbose=0,  # enables the training progress to be printed during the learning process
                         gamma=0.7,  # determines the importance of future rewards compared to immediate rewards
                         n_steps=200,  # steps to collect samples from the environment before performing an update
                         ent_coef=0.01,  # encourages exploration by adding entropy to the policy loss
                         learning_rate=0.001,  # controls the step size at which model's parameters are updated based on the gradient of the loss function
                         clip_range=0.1,  # limits the update to a certain range to prevent large policy updates
                         batch_size=15,
                         device=DEVICE)
    print("Done")

    print("# Training model...")
    model = model.learn(total_timesteps=N_TIMESTEPS)
    save_dir = './saved_models/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir + 'RecurrentPPO-' + datetime.isoformat(datetime.today()).split(".")[0].replace(":", "-")
    model.save(save_path)
    print(f"\n\nDone. Model saved at {save_path}\n\n")
else:
    print("# Loading model...")
    model = RecurrentPPO.load(MODEL_PATH)

# Calculate indicators for testing dataset

ind_path = './data/indicators.csv'
testing_df.to_csv(ind_path)
indicators = Indicators(ind_path)
indicators.to_csv(ind_path)
testing_df = pd.read_csv(ind_path)
testing_df["date_open"] = pd.to_datetime(testing_df["date_open"])
testing_df.set_index("date_open", inplace=True)
testing_df.dropna(inplace=True)


testing_env = gym.make("TradingEnv",
                       name="BTCUSD",
                       df=testing_df,
                       positions=[0, 1],  # 0(=SELL ALL), +1 (=BUY ALL)
                       portfolio_initial_value=1000,
                       initial_position=0)
add_env_metrics(testing_env)
observation, info = testing_env.reset()

print("# Testing model on testing data...")
for _ in range(len(testing_df)):
    position_index, _states = model.predict(observation)
    observation, reward, done, truncated, info = testing_env.step(position_index)
    if done or truncated:
        break
testing_env.save_for_render(dir=RENDER_DIR)
print(f"Test finished. Render saved in {RENDER_DIR}")

if RENDER_RESULTS:
    print("# Render results...")
    render_saved_logs(RENDER_DIR)
