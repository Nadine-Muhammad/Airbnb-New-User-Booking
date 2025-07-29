import torch
import torch.nn as nn
from src.config import config_handler
from src.const import constants as const

config = config_handler.load_config()

#Model architecture

class AirbnbNN(nn.Module):
    def __init__(self, input_size=config[const.MODEL][const.INPUT_SIZE],
                       hidden_size=config[const.MODEL][const.HIDDEN_SIZE],
                       output_size=config[const.MODEL][const.OUTPUT_SIZE]):
        super(AirbnbNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x