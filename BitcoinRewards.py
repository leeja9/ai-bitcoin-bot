import pandas as pd
import numpy as np
from gym_trading_env.utils.history import History


def reward_function(history: History, win_size: int = 144) -> float:
    """Multi-objective reward function

    Args:
        history (History): History object
        win_size (int, optional): window size. Defaults to -144 (6 days).

    Returns:
        float: Average of alr, sr, powc
    """
    # calculate log_returns for rolling window
    data_close = np.array(history['data_close'][-win_size:], dtype=np.float64)
    pos = np.array(history['position'][-win_size:], dtype=np.float64)
    log_data = np.log(data_close)
    log_return = np.where(pos[1:] == 1, np.diff(log_data), 0)
    if len(log_return) > 0:
        # average log return
        alr = log_return.mean()
        # sharpe ratio
        sr = 0 if log_return.std() == 0 else alr / log_return.std()
    else:
        alr = 0
        sr = 0
    # profit only when closed
    buy_pos = np.where(np.diff(pos) > 0)[0]
    sell_pos = np.where(np.diff(pos) < 0)[0]
    if len(buy_pos) == 0 or len(sell_pos) == 0:
        powc = 0
    else:
        if sell_pos[0] < buy_pos[0]:
            sell_pos = sell_pos[1:]
        log_profits = np.zeros(len(sell_pos))
        for i, b, s in zip(range(len(sell_pos)), buy_pos, sell_pos):
            log_profits[i] = log_return[s] - log_return[b]
        powc = log_profits.mean()
    reward_vector = np.array([alr, sr, powc])
    # the weights of a weighted average could be another parameter to optimize
    return reward_vector.mean()
