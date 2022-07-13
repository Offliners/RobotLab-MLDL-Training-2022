import torch.nn as nn

class Neural_Net(nn.Module):
    def __init__(self, input_dim):
        super(Neural_Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)

        return x