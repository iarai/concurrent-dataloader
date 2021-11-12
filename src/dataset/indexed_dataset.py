import abc
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class IndexedDataset(Dataset, abc.ABC):
    def __init__(self, index_file: Path, classes_file: Path):
        self.index_file = index_file
        self.classes_file = classes_file
        self.load_index()
        self.load_classes()

    @abc.abstractstaticmethod
    def index_all(self, **kwargs) -> None:
        raise NotImplementedError()

    def load_index(self) -> None:
        with self.index_file.open("r") as file:
            self.image_paths = json.load(file)

    def load_classes(self) -> None:
        with self.classes_file.open("r") as file:
            self.classes = json.load(file)

    # TODO should this be part of sampler instead of dataset?
    @abc.abstractmethod
    def get_random_item(self) -> Image:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_transform(self, transform: transforms) -> None:
        raise NotImplementedError()
