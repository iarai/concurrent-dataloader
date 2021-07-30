import json
import re
import random

import boto3
from PIL import Image
from torch.utils.data.dataset import Dataset
from io import BytesIO

FILENAME = "file_index.json"


class S3DataLoader(Dataset):
    def __init__(self, mode, bucket_name, access_key, secret_key):
        self.image_paths = []
        self.mode = "scratch/imagenet/" + mode

        self.bucket_name = bucket_name
        self.s3_client = boto3.resource(
            "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )

    def index_all(self):
        self.image_paths = list(
            self.s3_client.Bucket(self.bucket_name).objects.filter(Prefix=self.mode)
        )

    def save_index(self):
        str_paths = []
        [str_paths.append(str(i)) for i in self.image_paths]
        with open("index.json", "w") as file:
            json.dump(str_paths, file)

    def load_index(self):
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

    def get_random_item(self):
        rn = random.randint(0, self.__len__())
        image = self.__getitem__(rn)
        return image

    def __getitem__(self, index):
        b = BytesIO()
        self.s3_client.Bucket(self.bucket_name).download_fileobj(
            self.image_paths[index].key, b
        )
        image = Image.open(b)
        return image

    def __len__(self):
        return len(self.image_paths)
