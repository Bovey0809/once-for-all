import os
from torchvision.datasets import CIFAR10
from .imagenet import ImagenetDataProvider
from torchvision import transforms


class Cifar10DataProvider(ImagenetDataProvider):
    DEFAULT_PATH = "./data/cifar10"
    DEFAULT_IMAGE_SIZE = 32

    @staticmethod
    def name():
        return "cifar10"

    @property
    def n_classes(self):
        return 10  # CIFAR-10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/data/cifar10")
        return self._save_path

    @property
    def data_url(self):
        return "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )

    def train_dataset(self, _transforms):
        return CIFAR10(
            root=self.save_path, train=True, download=True, transform=_transforms
        )

    def test_dataset(self, _transforms):
        return CIFAR10(
            root=self.save_path, train=False, download=True, transform=_transforms
        )
