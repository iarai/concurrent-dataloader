import importlib
import json
import re
from io import BytesIO

from PIL import Image
from misc.random_generator import RandomGenerator
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from misc.time_helper import stopwatch
import importlib


class S3Dataset(Dataset):
    def __init__(
            self,
            mode: str,
            bucket_name: str,
            limit: int = None,
    ) -> None:
        self.limit = limit
        self.mode = "scratch/imagenet/" + mode
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
        self.bucket_name = bucket_name
        self.rng = RandomGenerator()
        self.pickle_buffer = BytesIO()

        boto_module = importlib.import_module("dataset.boto_mediator")
        self.len = len(boto_module.image_paths)


    def index_all(self) -> None:
        pass

    def save_index(self) -> None:
        boto_module = importlib.import_module("dataset.boto_mediator")
        boto_module.save_index()

    def load_index(self) -> None:
        pass

    def get_random_item(self) -> Image:
        rn = self.rng.get_int(0, self.__len__())
        return self.__getitem__(rn)

    @stopwatch("(5)-get_item")
    def __getitem__(self, index: int) -> Image:
        boto_module = importlib.import_module("dataset.boto_mediator")
        image = boto_module.getitem(index)
        return self.transform(image)

    def __len__(self):
        return self.len
