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
    # n_history = len(history['data_close'])
    # if  > n_history:
    #     win_size = -n_history
    data_close = pd.Series(history['data_close'][-win_size:])
    log_return = data_close.apply(lambda x: np.log(x)).pct_change()
    log_return = log_return.fillna(0)
    average_log_return = log_return.mean()
    std_dev = log_return.std()
    # risk_free_rate = 0.025  # 2.5%
    sharpe_ratio = (average_log_return) / std_dev
    positions = pd.Series(history['position'][-win_size:])
    powc = log_return.mask(positions == 0, 0).mean()  # profit only when closed
    reward_vector = np.array([powc, average_log_return, sharpe_ratio])
    # the weights of a weighted average could be another parameter to optimize
    return reward_vector.mean()
