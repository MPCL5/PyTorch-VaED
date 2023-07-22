import torch
import torch.nn as nn

from model.decoder import Decoder
from model.encoder import Encoder


class VaED(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, latent_dim: int, n_centroid: int, alpha: int, datatype='sigmoid') -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.n_centroid = n_centroid
        self.alpha = alpha
        self.datatype = datatype
        self.latent_dim = latent_dim

        # GMM parameters
        # TODO: check whether if this snippest can be extracted.
        self.theta_p = nn.Parameter(torch.rand(n_centroid))
        self.u_p = nn.Parameter(torch.randn(latent_dim, n_centroid))
        self.lambda_p = nn.Parameter(torch.rand(latent_dim, n_centroid))

    def get_gamma(self, z):
        temp_Z = z.unsqueeze(2).repeat(1, 1, self.n_centroid)
        temp_u_tensor3 = self.u_p.unsqueeze(0).repeat(temp_Z.size(0), 1, 1)
        temp_lambda_tensor3 = self.lambda_p.unsqueeze(
            0).repeat(temp_Z.size(0), 1, 1)
        temp_theta_tensor3 = self.theta_p.unsqueeze(0).unsqueeze(
            0) * torch.ones((temp_Z.size(0), temp_Z.size(1), self.n_centroid), device=next(self.parameters()).device)

        p_c_z = torch.exp(torch.sum((torch.log(temp_theta_tensor3) - 0.5 * torch.log(2 * torch.pi * temp_lambda_tensor3) -
                                     torch.square(temp_Z - temp_u_tensor3) / (2 * temp_lambda_tensor3)), dim=1)) + 1e-10

        gamma = p_c_z / torch.sum(p_c_z, dim=-1, keepdim=True)
        return gamma

    def get_gamma_with_x(self, x):
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        return self.get_gamma(z)

    def vae_loss(self, x, x_decoded_mean):
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        gamma = self.get_gamma(z)
        gamma_t = gamma.unsqueeze(1).repeat(1, self.latent_dim, 1)
        z_mean_t = mu_e.unsqueeze(2).repeat(1, 1, self.n_centroid)
        z_log_var_t = log_var_e.unsqueeze(2).repeat(1, 1, self.n_centroid)
        u_tensor3 = self.u_p.unsqueeze(0).repeat(z.size(0), 1, 1)
        lambda_tensor3 = self.lambda_p.unsqueeze(0).repeat(z.size(0), 1, 1)

        if self.datatype == 'sigmoid':
            loss = self.alpha * x.shape[-1] * nn.BCELoss(reduction='sum')(x_decoded_mean, x)\
                + torch.sum(0.5 * gamma_t * (self.latent_dim * torch.log(torch.tensor(torch.pi * 2)) + torch.log(lambda_tensor3)
                                             + torch.exp(z_log_var_t) / lambda_tensor3 + torch.square(z_mean_t - u_tensor3) / lambda_tensor3), dim=(1, 2))\
                - 0.5 * torch.sum(log_var_e+1, dim=-1)\
                - torch.sum(torch.log(self.theta_p.unsqueeze(0).repeat(x.shape[0], 1)) * gamma, dim=-1)\
                + torch.sum(torch.log(gamma) * gamma, dim=-1)
        else:
            loss = self.alpha * x.shape[-1] * nn.MSELoss(reduction='sum')(x, x_decoded_mean)\
                + torch.sum(0.5 * gamma_t * (self.latent_dim * torch.log(torch.pi * 2) + torch.log(lambda_tensor3)
                                             + torch.exp(z_log_var_t) / lambda_tensor3 + torch.square(z_mean_t - u_tensor3) / lambda_tensor3), dim=(1, 2))\
                - 0.5 * torch.sum(log_var_e+1, dim=-1)\
                - torch.sum(torch.log(self.theta_p.unsqueeze(0).repeat(x.shape[0], 1)) * gamma, dim=-1)\
                + torch.sum(torch.log(gamma)*gamma, dim=-1)

        return loss.mean()

    def forward(self, x):
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        x_decoded_mean = self.decoder.decode(z)
        return x_decoded_mean, mu_e, log_var_e
