import pandas as pd
import numpy as np
from gym_trading_env.utils.history import History

# TODO: REFACTOR


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
        this_lr = np.log(history['data_close', -1]) - np.log(history['data_close', -2])
    history.__setitem__(('data_lr', -1), this_lr)  # update history with new lr

    # running average of logarithmic return
    n = len(history)
    last_alr = history['data_alr', -2]
    this_alr = ((n - 1) * last_alr + this_lr) / n
    history.__setitem__(('data_alr', -1), this_alr)  # update history with new alr

    # running variance sum of logarithmic return
    # for each nth row, dividing this sum by n gives population variance
    last_alr = history['data_alr', -2]
    last_var_sum = history['data_var_sum', -2]
    this_var_sum = last_var_sum + abs((this_lr - last_alr) * (this_lr - this_alr))
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
    sharpe_ratio = average_log_return / std_dev
    this_lr = history['data_lr', -1]
    powc = 0
    # if this eposide position is 0 (100% USD) and last position was 1 (100% BTC)
    # this compute time can also be traded for memory by adding a tracking column if needed
    if (history['position', -1] == 0 and history['position', -2] == 1):
        idx = history[-2]['idx']
        while idx >= 0:
            if (history['position', idx] == 0):
                last_lr = history['data_lr', idx + 1]
                powc = this_lr - last_lr
    reward_vector = np.array([average_log_return, sharpe_ratio, powc])
    weight_vector = get_random_weights(len(reward_vector))
    reward = reward_vector @ weight_vector  # dot product of random weights and reward values
    return reward
# def add_delta_pct_change(df: pd.DataFrame):
#     """natural log percent change of closing prices"""
#     df["log_pct_change"] = df["close"].apply(np.log).pct_change()
#     df["log_delta"] = df["close"].apply(np.log).diff()
# add_delta_pct_change(df)
# print(df["close"].head(), df["log_delta"].head())
