import pandas as pd
import numpy as np
from gym_trading_env.utils.history import History
from gym_trading_env.renderer import Renderer
from gym_trading_env.environments import TradingEnv


def get_buy_i(history: History) -> np.ndarray:
    p = np.array(history['position'])
    return np.where(np.diff(p) > 0)[0] + 1


def get_sell_i(history: History) -> np.ndarray:
    p = np.array(history['position'])
    return np.where(np.diff(p) < 0)[0] + 1


def get_pct_changes(history: History) -> np.ndarray:
    v = history['portfolio_valuation']
    buys = get_buy_i(history)
    sells = get_sell_i(history)
    if sells[0] < buys[0]:
        sells = sells[1:]
    return np.array(
        [(v[sell] / v[buy] - 1) * 100 for buy, sell in zip(buys, sells)]
    )


def max_drawdown(history: History) -> str:
    pct_changes = get_pct_changes(history)
    m = pct_changes.min()
    return f"{round(m, 5)}%"


def max_gain(history: History) -> str:
    pct_changes = get_pct_changes(history)
    m = pct_changes.max()
    return f"{round(m, 5)}%"


def n_win(history: History) -> str:
    """number of position changes with positive or neutral valuation change"""
    pct_changes = get_pct_changes(history)
    return str(len(np.where(pct_changes >= 0)[0]))


def n_loss(history: History) -> str:
    """number of position changes with negative valuation change"""
    pct_changes = get_pct_changes(history)
    return str(len(np.where(pct_changes < 0)[0]))


def avg_win(history: History) -> str:
    """average percent win"""
    pct_changes = get_pct_changes(history)
    wins = np.where(pct_changes >= 0, pct_changes, 0).sum() / len(np.where(pct_changes >= 0)[0])
    return f"{round(wins, 5)}%"


def avg_loss(history: History) -> str:
    """average percent loss"""
    pct_changes = get_pct_changes(history)
    losses = np.where(pct_changes < 0, pct_changes, 0).sum() / len(np.where(pct_changes < 0)[0])
    return f"{round(losses, 5)}%"


def add_render_metrics(renderer: Renderer) -> None:
    renderer.add_metric(name='Max Drawdown', function=max_drawdown)
    renderer.add_metric(name='Max Gain', function=max_gain)
    renderer.add_metric(name='Num Win', function=n_win)
    renderer.add_metric(name="Avg Win", function=avg_win)
    renderer.add_metric(name="Num Loss", function=n_loss)
    renderer.add_metric(name="Avg Loss", function=avg_loss)
    renderer.add_line(name="144hr Avg Close", function=lambda df: df['close'].rolling(144).mean(), line_options ={"width" : 1, "color": "blue"})


def add_env_metrics(env: TradingEnv) -> None:
    env.add_metric('Max Drawdown', max_drawdown)
    env.add_metric('Max Gain', max_gain)
    env.add_metric('Num Win', n_win)
    env.add_metric("Avg Win", avg_win)
    env.add_metric("Num Loss", n_loss)
    env.add_metric("Avg Loss", avg_loss)
