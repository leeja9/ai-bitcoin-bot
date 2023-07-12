from BitcoinTrainingActions import BitcoinTrainingActions


class BitcoinTrainingAgent:
    actions = BitcoinTrainingActions()

    def __init__(
            self    # TODO define hyperparameters (rates, epsilon, etc.)
    ) -> None:

        # TODO instantiate passed hyperparameters
        pass

    # PRIVATE METHODS
    # TODO: Implement methods
    #       1. Update state based on ML algorithm
    #       2. Other helper functions as necessary
    def _update_state(self):
        # TODO
        pass

    # PUBLIC METHODS
    # TODO: Implement methods
    #       1. Provide actions taken after state change (consumed by env)

    def get_actions(self):
        # TODO: Define action based on current state
        return self.actions
