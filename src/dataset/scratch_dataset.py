import json
import os
from pathlib import Path
from typing import Optional

from torch.functional import split

from dataset.indexed_dataset import IndexedDataset
from misc.random_generator import RandomGenerator
from misc.time_helper import stopwatch
from overrides import overrides
from PIL import Image
from torchvision import transforms


class ScratchDataset(IndexedDataset):
    def __init__(self, index_file: Path, classes_file: Optional[Path] = None, limit: int = None) -> None:
        self.limit = limit
        self.classes_file = classes_file
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])
        self.rng = RandomGenerator()
        super().__init__(index_file=index_file, classes_file=classes_file)


    @staticmethod
    def index_all(imagenet_path: Path, file_name: Optional[str], **kwargs) -> None:
        # TODO magic string constants JPEG -> extract as param
        image_paths = list(imagenet_path.rglob("**/*.JPEG"))
        if file_name is not None:
            with open(file_name, "w") as file:
                json.dump([(str(i)) for i in image_paths], file)

    @overrides
    def set_transform(self, transform: transforms) -> None:
        self.transform = transform

    @overrides
    def get_random_item(self) -> Image:
        rn = self.rng.get_int(0, self.__len__() - 1)
        return self.__getitem__(rn)

    # TODO we should do the do the @stopwatch instrumentalization only in the benchmarking part and
    #  keep this code clean from those aspects! Decorator for decorator...?
    # TODO we should make this code independent of the underlying dataset, not necessarily images/imagenet?
    @stopwatch(trace_name="(5)-get_item", trace_level=5, strip_result=True)
    def __getitem__(self, index) -> Image:
        class_folder_name = self.image_paths[index].split("/")[4]        
        if self.classes is not None:
            # validation dataset
            if class_folder_name.startswith("ILSV"):
                class_folder_name = int(class_folder_name.replace(".JPEG", "").split("_")[2])
                target = self.classes[str(class_folder_name)]
            # target dataset
            elif class_folder_name.startswith("n"):
                target = self.classes[class_folder_name]["id"]
            else:
                raise ValueError(
                    "Unexpected file name. Training image names should start with 'n', while"
                    " validation image paths should start with 'ILSV'."
                )
        else:
            target = None

        image_path = self.image_paths[index]
        image = Image.open(image_path)

        image_path = self.image_paths[index]
        image = Image.open(image_path)

        # some images in the scratch dataset seem to be in "L" instead of "RGB" mode 
        if image.mode == "L":
            image = image.convert("RGB")

        image = self.transform(image)
        return image, target, os.path.getsize(image_path)

    def __len__(self) -> int:
        if self.limit is None:
            return len(self.image_paths)
        else:
            return len(self.image_paths[: self.limit])