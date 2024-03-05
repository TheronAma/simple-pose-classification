import torch
import numpy as np

class PoseClassifier(torch.nn.Module):

    def __init__(self,
            input_size=34,
            hidden_size=256,
            output_size=3,
            nonlinearity=torch.nn.GELU):

        super(PoseClassifier, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            nonlinearity(),
            torch.nn.Linear(hidden_size, hidden_size),
            nonlinearity(),
            torch.nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out = self.model(x)
        return out
