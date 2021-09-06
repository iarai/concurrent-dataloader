import json
from pathlib import Path
from typing import Optional

from dataset.indexed_dataset import IndexedDataset
from misc.random_generator import RandomGenerator
from misc.time_helper import stopwatch
from overrides import overrides
from PIL import Image
from torchvision import transforms


class ScratchDataset(IndexedDataset):
    def __init__(self, index_file: Path, limit: int = None) -> None:
        super().__init__(index_file=index_file)
        self.limit = limit
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])
        self.rng = RandomGenerator()

    @staticmethod
    def index_all(imagenet_path: Path, file_name: Optional[str], **kwargs) -> None:
        # TODO magic string constants JPEG -> extract as param
        image_paths = list(imagenet_path.rglob("**/*.JPEG"))
        if file_name is not None:
            with open(file_name, "w") as file:
                json.dump([(str(i)) for i in image_paths], file)

    @overrides
    def get_random_item(self) -> Image:
        rn = self.rng.get_int(0, self.__len__() - 1)
        return self.__getitem__(rn)

    # TODO we should do the do the @stopwatch instrumentalization only in the benchmarking part and
    #  keep this code clean from those aspects! Decorator for decorator...?
    # TODO we should make this code independent of the underlying dataset, not necessarily images/imagenet?
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
