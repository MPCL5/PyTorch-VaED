import os
from typing import Callable
from pyparsing import Any
import requests
from scipy.io import loadmat
from torch.utils.data import Dataset

DOWNLOAD_LINK = "https://raw.githubusercontent.com/slim1017/VaDE/master/dataset/har/HAR.mat"


class HarDataset(Dataset):
    def __init__(self, root, train=True, transform: Callable[..., Any] |
                 None = None, target_transform: Callable[..., Any] |
                 None = None, download: bool = False) -> None:
        super(Dataset, self).__init__()
        file_path = os.path.join(root, 'HAR.mat')

        if not os.path.exists(file_path):
            print(f"Could not find HAR.mat at {root}")

            if not download:
                raise Exception('Please download the dataset.')

            self.__donwload(file_path)

        self.data = loadmat(file_path)
        self.data['X'] = self.data['X'].astype('float32')
        self.data['Y'] = self.data['Y'] - 1
        self.transform = transform
        self.target_transform = target_transform

    def __donwload(self, path):
        print('Downloading HAR dataset')
        data = requests.get(DOWNLOAD_LINK, allow_redirects=True)

        open(path, 'wb').write(data.content)

    def __len__(self):
        return len(self.data['X'])

    def __getitem__(self, index):
        X, y = self.data['X'][index], self.data['Y'][index]

        if self.transform:
            X = self.transform(X)

        if self.target_transform:
            y = self.target_transform(y)

        return (X, y)


if __name__ == "__main__":
    test_data = HarDataset(root='./data/har', download=True)

    print(test_data.__len__())
    print(type(test_data[0][1]))
