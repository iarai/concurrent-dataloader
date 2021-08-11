import json
import re
from io import BytesIO

import boto3
from misc.random_generator import RandomGenerator
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class S3Dataset(Dataset):
    def __init__(self, mode: str, bucket_name: str, access_key: str, secret_key: str) -> None:
        self.image_paths = []
        self.mode = "scratch/imagenet/" + mode
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])
        self.bucket_name = bucket_name
        self.s3_client = boto3.resource("s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        self.rng = RandomGenerator()

    def index_all(self) -> None:
        self.image_paths = list(self.s3_client.Bucket(self.bucket_name).objects.filter(Prefix=self.mode))

    def save_index(self) -> None:
        str_paths = []
        [str_paths.append(str(i)) for i in self.image_paths]
        with open("index.json", "w") as file:
            json.dump(str_paths, file)

    def load_index(self) -> None:
        str_paths = None
        with open("index.json", "r") as file:
            str_paths = json.load(file)
        if self.image_paths is not None:
            self.image_paths.clear()

        # todo: find a cleverer way to do this (or another approach completley)
        for i in str_paths:
            [i := i.replace(j, "") for j in ",')"]
            i = re.findall(r'(\S+)=(".*?"|\S+)', i)
            self.image_paths.append(self.s3_client.ObjectSummary(i[0][1], i[1][1]))

    def get_random_item(self) -> Image:
        rn = self.rng.get_int(0, self.__len__())
        return self.__getitem__(rn)

    def __getitem__(self, index: int) -> Image:
        b = BytesIO()
        self.s3_client.Bucket(self.bucket_name).download_fileobj(self.image_paths[index].key, b)
        image = Image.open(b)
        return self.transform(image)

    def __len__(self):
        return len(self.image_paths)
