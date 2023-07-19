import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_model_summary import summary
import torchvision.transforms as T

from dataset.mnist import MnistDataset
from model.decoder import Decoder
from model.encoder import Encoder
from model.vaed import VaED
from utils.train_supervisor import TrainSupervisor
from utils.trainer import Trainer


MODEL_NAME = 'VaED'
IMG_SIZE = 28  # input dimension
BATCH_SIZE = 32
L = 10  # number of latents
M = 256  # the number of neurons in scale (s) and translation (t) nets

D_TYPE = 'sigmoid'
ALPHA = 0.9  # alpha: a hyper param!?
LR = 5e-4  # learning rate
NUM_EPOCHS = 1000  # max. number of epochs
MAX_PATIENCE = 20  # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
NUM_CHANNELS = 3
NUM_CENTROIDS = 10
NUM_VALS = 256

DEFAULT_DATA_SET = 'mnist'
RESULT_DIR = './results/'

GENERATED_NUM_X = 10  # number of images in the X axis of generated result
GENERATED_NUM_Y = 10  # number of images in the Y axis of generated result

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_structure():
    if (os.path.exists(RESULT_DIR)):
        return

    os.mkdir(RESULT_DIR)


def get_mnist_loader():
    img_transform = T.Compose([
        T.ToTensor(),
    ])

    train_data = MnistDataset(
        root='./data/mnist', download=True, transform=img_transform)
    validation_data = MnistDataset(
        root='./data/mnist', download=True, train=False, transform=img_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(
        validation_data, batch_size=BATCH_SIZE, shuffle=True)

    return (train_loader, validation_loader)


def get_data_loaders(dataset='mnist'):
    match dataset:
        case 'mnist':
            return get_mnist_loader()

        case _:
            raise ValueError('Unknown dataset.')

def data_transformer(data):
    data = data.to(DEVICE)
    data = data.view(data.shape[0], IMG_SIZE * IMG_SIZE)

    return data

def get_model():
    encoder_net = nn.Sequential(
        nn.Linear(IMG_SIZE * IMG_SIZE, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, 2000),
        nn.ReLU(),
        nn.Linear(2000, L * 2),
        nn.ReLU(),
    ).to(DEVICE)

    decoder_net = nn.Sequential(
        nn.Linear(L, 2000),
        nn.ReLU(),
        nn.Linear(2000, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, IMG_SIZE * IMG_SIZE),
        nn.Sigmoid(),
    ).to(DEVICE)

    encoder = Encoder(encoder_net).to(DEVICE)
    decoder = Decoder(decoder_net).to(DEVICE)

    vaed = VaED(encoder=encoder, decoder=decoder, latent_dim=L,
                n_centroid=NUM_CENTROIDS, alpha=ALPHA, datatype=D_TYPE).to(DEVICE)

    # TODO: extract the summary logic from this code.
    print("ENCODER:\n", summary(encoder, torch.zeros(1, IMG_SIZE * IMG_SIZE,
          device=DEVICE), show_input=False, show_hierarchical=False))

    print("\nDECODER:\n", summary(decoder, torch.zeros(
        1, L, device=DEVICE), show_input=False, show_hierarchical=False))

    print("\nVaED:\n", summary(vaed, torch.zeros(
        1, IMG_SIZE * IMG_SIZE, device=DEVICE), show_input=True, show_hierarchical=True))

    return vaed


def get_optimizer():
    return torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=LR)


def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    ensure_structure()
    train_loader, validation_loader = get_data_loaders(DEFAULT_DATA_SET)

    model = get_model()
    optimizer = get_optimizer()

    supervisor = TrainSupervisor(MODEL_NAME, MAX_PATIENCE, RESULT_DIR)
    trainer = Trainer(supervisor, NUM_EPOCHS, model,
                      optimizer, train_loader, validation_loader, data_transformer)

    nll_val = trainer.train()
    print(nll_val)
    plot_curve(RESULT_DIR + MODEL_NAME, nll_val)
