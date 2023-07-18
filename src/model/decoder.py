import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()

        self.network = network
        
    def decode(self, z):
        return self.network(z)

    def forward(self, x):
        return self.decode(x)
