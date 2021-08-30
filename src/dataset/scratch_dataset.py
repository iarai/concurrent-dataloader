import json
import os
from pathlib import Path
from typing import Optional

from dataset.indexed_dataset import IndexedDataset
from misc.random_generator import RandomGenerator
from misc.time_helper import stopwatch
from overrides import overrides
from PIL import Image
from torchvision import transforms

# TODO magic string constants in code
IMAGENET_PATH_SCRATCH = "/scratch/imagenet"


class ScratchDataset(IndexedDataset):
    def __init__(self, imagenet_path: str, mode: str, limit: int = None) -> None:
        self.mode = mode
        self.limit = limit
        self.image_paths = []
        assert mode in ["train", "val"], mode
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])
        self.rng = RandomGenerator()
        self.__imagenet_path = Path(os.path.join(imagenet_path, mode))

    def index_all(self, file_name: Optional[str], **kwargs) -> None:
        # TODO magic string constants JPEG -> extract as param
        self.image_paths = list(self.__imagenet_path.rglob("**/*.JPEG"))
        if file_name is not None:
            with open(file_name, "w") as file:
                json.dump([(str(i)) for i in self.image_paths], file)

    @overrides
    def get_random_item(self) -> Image:
        rn = self.rng.get_int(0, self.__len__() - 1)
        return self.__getitem__(rn)

    @stopwatch("(5)-get_item")
    def __getitem__(self, index) -> Image:
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        return self.transform(image)

    def __len__(self) -> int:
        if self.limit is None:
            return len(self.image_paths)
        else:
            return len(self.image_paths[: self.limit])
