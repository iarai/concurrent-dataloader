import json
import os
import random
from pathlib import Path

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

IMAGENET_PATH_SCRATCH = "/scratch/imagenet"


class ScratchDataset(Dataset):
    def __init__(self, imagenet_path: str, mode: str) -> None:
        self.mode = mode
        self.image_paths = []
        assert mode in ["train", "val"], mode
        self.transform =  transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])

        self.__imagenet_path = Path(os.path.join(imagenet_path, mode))

    def index_all(self) -> None:
        self.image_paths = list(self.__imagenet_path.rglob("**/*.JPEG"))

    def save_index(self) -> None:
        str_paths = []
        [str_paths.append(str(i)) for i in self.image_paths]
        with open("index.json", "w") as file:
            json.dump(str_paths, file)

    def load_index(self) -> None:
        __str_paths = None
        with open("index.json", "r") as file:
            __str_paths = json.load(file)
        if self.image_paths is not None:
            self.image_paths.clear()
        [self.image_paths.append(Path(i)) for i in __str_paths]

    def get_random_item(self) -> Image:
        rn = random.randint(0, self.__len__() - 1)
        return self.__getitem__(rn)

    def __getitem__(self, index) -> Image:
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        return self.transform(image)

    def __len__(self) -> int:
        return len(self.image_paths)
