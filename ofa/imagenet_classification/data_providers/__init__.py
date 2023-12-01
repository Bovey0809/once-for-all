# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from .imagenet import ImagenetDataProvider
from .cifar10 import Cifar10DataProvider

__all__ = ['ImagenetDataProvider', 'Cifar10DataProvider']