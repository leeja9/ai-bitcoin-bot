import pandas as pd
import numpy as np
from gym_trading_env.utils.history import History


def dynamic_feature_last_position_taken(history):
    return history['position', -1]


def dynamic_feature_real_position(history):
    return history['real_position', -1]
