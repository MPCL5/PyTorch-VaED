import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.cluster import cluster_acc

from utils.train_supervisor import TrainSupervisor


class Trainer:
    # TODO: Too many dependencies. should be refactored!
    def __init__(self, supervisor: TrainSupervisor, num_epochs: int,
                 model: nn.Module, optimizer, training_loader: DataLoader, val_loader: DataLoader, data_transfer=None) -> None:
        self.num_epochs = num_epochs
        self.model = model
        self.optimizer = optimizer
        self.training_loader = training_loader
        self.val_loader = val_loader
        self.data_transfer = data_transfer
        self.supervisor = supervisor

    def __train_once(self):
        self.model.train()  # put our model in train mode
        for _, (batch, _) in enumerate(self.training_loader):
            if self.data_transfer:
                batch = self.data_transfer(batch)

            (x_decoded_mean, _, _) = self.model.forward(batch)
            loss = self.model.vae_loss(batch, x_decoded_mean)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def evaluate(self, epoch):
        self.model.eval()
        loss = 0.
        N = 0.
        acc = 0.

        for _, (test_batch, target) in enumerate(self.val_loader):
            if self.data_transfer:
                test_batch = self.data_transfer(test_batch)

            gamma = self.model.get_gamma_with_x(test_batch)
            acc += cluster_acc(torch.argmax(gamma,
                               dim=1).cpu().numpy(), target.cpu().numpy())

            (x_decoded_mean, _, _) = self.model.forward(test_batch)
            loss_t = self.model.vae_loss(test_batch, x_decoded_mean)
            loss = loss + loss_t.item()
            N = N + test_batch.shape[0]

        loss = loss / N
        acc = acc / N

        print(
            f'Epoch: {epoch if epoch is not None else "Final"}, val nll={loss}, cluster acc={acc}'
        )
        # print((self.model.theta_p, self.model.u_p, self.model.lambda_p))

        return loss

    def train(self):
        nll_val = []
        self.supervisor.set_model(self.model)

        for e in range(self.num_epochs):
            self.__train_once()
            loss_val = self.evaluate(e)

            nll_val.append(loss_val)  # save for plotting
            self.supervisor.proceed(loss_val)

            if self.supervisor.is_breakable():
                break

        nll_val = np.asarray(nll_val)

        return nll_val
