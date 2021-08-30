import abc

from PIL import Image
from torch.utils.data import Dataset


class IndexedDataset(Dataset, abc.ABC):
    def __init__(self):
        self.image_paths = self.load_index()

    @abc.abstractstaticmethod
    def index_all(self, **kwargs) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def load_index(self) -> None:
        raise NotImplementedError()

    # TODO should this be part of sampler instead of dataset?
    @abc.abstractmethod
    def get_random_item(self) -> Image:
        raise NotImplementedError()
