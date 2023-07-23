# based on FiRe repository actions_and_positions.py
# https://github.com/trality/fire
from enum import Enum


class BitcoinTrainingActions(Enum):
    Buy = 0
    Hold = 1
    Sell = 2


class BitcoinTrainingPositions(Enum):
    Buy = 0
    Hold = 1
    Sell = 2
