# Description: TODO

from BitcoinTrainingFilters import BitcoinTrainingFilters
from BitcoinTrainingStates import BitcoinTrainingStates


class BitcoinTrainingEnvironment:
    def __init__(self) -> None:
        pass

    # PRIVATE METHODS
    # TODO: Implement methods
    #       1. Retreive data (fixed file, steady stream, combination?)
    #       2. Filter
    #       3. Create states based on filtered data (and actions?)
    #       4. Retrieve ML actions
    #       5. Update environment after episode completes
    #       6. Other helper functions as needed
    def _get_data(self):
        # TODO
        pass

    def _get_filter(self):
        # TODO
        filters = BitcoinTrainingFilters()
        return filters

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
