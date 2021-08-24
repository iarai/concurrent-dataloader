import importlib
import json
import re
from io import BytesIO

from PIL import Image
from misc.random_generator import RandomGenerator
from torch.utils.data.dataset import Dataset
from torchvision import transforms
# from dataset import bototo
from misc.time_helper import stopwatch
import importlib


class S3Dataset(Dataset):
    def __init__(
            self,
            mode: str,
            bucket_name: str,
            limit: int = None,
    ) -> None:
        self.image_paths = []
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
        # self.s3_client = importlib.import_module("dataset.bototo")
        print(bucket_name)
        li = importlib.import_module("dataset.bototo")
        self.len = len(li.image_paths)

    def index_all(self) -> None:
        # self.image_paths = list(bototo.s3_client.Bucket(self.bucket_name).objects.filter(Prefix=self.mode))
        pass

    def save_index(self) -> None:
        str_paths = []
        [str_paths.append(str(i)) for i in self.image_paths]
        with open("index-s3.json", "w") as file:
            json.dump(str_paths, file)

    def load_index(self) -> None:
        pass
        # li = importlib.import_module("dataset.bototo").li() #getattr(__import__('dataset.bototo'), 'get_client')
        # self.image_paths = li.li()
        # str_paths = None
        # with open("index-s3.json", "r") as file:
        #     str_paths = json.load(file)
        # if self.image_paths is not None:
        #     self.image_paths.clear()
        #
        # # todo: find a cleverer way to do this (or another approach completley)
        # for i in str_paths:
        #     [i := i.replace(j, "") for j in ",')"]
        #     i = re.findall(r'(\S+)=(".*?"|\S+)', i)
        #     self.image_paths.append(s3_client.ObjectSummary(i[0][1], i[1][1]))

    def get_random_item(self) -> Image:
        rn = self.rng.get_int(0, self.__len__())
        return self.__getitem__(rn)

    @stopwatch("(5)-get_item")
    def __getitem__(self, index: int) -> Image:
        # b = BytesIO()
        # print(f"Index: {index}")
        # s3_client.Bucket(self.bucket_name).download_fileobj(self.image_paths[index].key, b)
        # image = Image.open(b)
        li = importlib.import_module("dataset.bototo")  # getattr(__import__('dataset.bototo'), 'get_client')
        image = li.getitem(index)
        return self.transform(image)

    def __len__(self):
        return self.len
        # if self.limit is None:
        #     return len(self.image_paths)
        # else:
        #     return len(self.image_paths[: self.limit])
