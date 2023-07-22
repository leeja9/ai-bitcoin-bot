# Description: TODO

# Gym API Reference: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

from BitcoinTrainingFilters import preprocess_bitcoin_ohlc_data

from BitcoinTrainingStates import BitcoinTrainingStates
from BitcoinTrainingRewardAgent import BitcoinTrainingRewardAgent
import gymnasium as gym
import pandas as pd


class BitcoinTrainingEnvironment(gym.Env):
    def __init__(self,
                 reward_agent: BitcoinTrainingRewardAgent,
                 data_path: str) -> None:
        self.preprocessed_data = self._get_data(data_path)

    # PRIVATE METHODS
    # TODO: Implement methods
    #       1. Retreive data (fixed file, steady stream, combination?)
    #       2. Filter
    #       3. Create states based on filtered data (and actions?)
    #       4. Retrieve ML actions
    #       5. Update environment after episode completes
    #       6. Other helper functions as needed
    def _get_data(self, data_path: str) -> pd.DataFrame:
        output_path = data_path.split('/')
        output_path = '/'.join(output_path[:len(output_path) - 1])
        output_path += 'preprocessed_output.csv'
        return preprocess_bitcoin_ohlc_data(data_path, output_path)

    def _get_states(self):
        # TODO
        states = BitcoinTrainingStates()
        return states

    def _update_env(self):
        pass

    # PUBLIC METHODS
    # TODO: Implement methods
    #       1. Methods consumed by observer
    #       2. Methods consumed by RewardAgent
    #       3. Any additional output interfaces not yet defined
