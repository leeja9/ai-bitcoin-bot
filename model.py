import torch
import torch.nn as nn


class BasicModel(nn.Module):
    """A PyTorch implementation of the model used in the trality/fire codebase.
    https://github.com/trality/fire/blob/master/src/model.py
    """

    def __init__(self,
                 input_size: int = 31,
                 scale_NN: int = 1,
                 dropout_level: float = 0,
                 number_outputs: int = 2,
                 ):
        super(BasicModel, self).__init__()

        self.dropout_level = dropout_level
        
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_level)

        self.fc2 = nn.Linear(512 * scale_NN, 256 * scale_NN)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_level)

        self.fc3 = nn.Linear(256 * scale_NN, 128 * scale_NN)
        self.relu3 = nn.ReLU()

        self.output_layer = nn.Linear(128 * scale_NN, number_outputs)

    def forward(
        self,
        price_signals,
        position,
        reward_weights,
        gamma
    ):
        # Flatten the price tensor into 2D.
        x = price_signals.view(price_signals.size(0), -1)
        # Concatenate all inputs into a single tensor.
        x = torch.cat((x, position, reward_weights, gamma), dim=1)

        x = self.fc1(x)
        x = self.relu1(x)
        if self.dropout_level > 0:
            x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        if self.dropout_level > 0:
            x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.output_layer(x)

        return x
