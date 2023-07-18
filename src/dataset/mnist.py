from typing import Callable
from pyparsing import Any
from torchvision.datasets import MNIST


class MnistDataset(MNIST):
    def __init__(self, root: str, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # we just need while we run this script directly

    test_data = MnistDataset(root='./data/mnist', download=True)

    print(test_data[0][0].size)
    plt.figure(figsize=(8, 8))
    plt.imshow(test_data[0][0])
    plt.show()
