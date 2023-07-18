import torch
import torch.nn as nn


class Encoder(nn.Module):
    @staticmethod
    def reparameterization(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + std * eps

    def __init__(self, network: nn.Module) -> None:
        super().__init__()

        self.network = network

    def encode(self, x):
        h_e = self.network(x)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        return mu_e, log_var_e

    def sample(self, mu_e=None, log_var_e=None):
        if (mu_e is None) or (log_var_e is None):
            raise ValueError('mu and log-var can`t be None!')

        z = Encoder.reparameterization(mu_e, log_var_e)
        return z

    def sample_with_x(self, x):
        mu_e, log_var_e = self.encode(x)

        return self.sample(mu_e, log_var_e)
    
    def forward(self, x):
        return self.sample_with_x(x)
