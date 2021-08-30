import importlib
import json
import os
import re
from io import BytesIO
from typing import Optional

from dataset.indexed_dataset import IndexedDataset
from misc.random_generator import RandomGenerator
from misc.time_helper import stopwatch
from overrides import overrides
from PIL import Image
from torchvision import transforms


def get_s3_cred(credentials_json="s3_credentials.json") -> json:
    __keys = None
    with open(credentials_json, "r") as file:
        __keys = json.load(file)
    return __keys


def get_s3_bucket(credentials_json):
    keys = get_s3_cred(credentials_json=credentials_json)
    boto3 = importlib.import_module("boto3")
    s3_client = boto3.resource(
        "s3",
        aws_access_key_id=keys["access_key"],
        aws_secret_access_key=keys["secret"],
        endpoint_url=keys.get("endpoint_url", "http://s3.amazonaws.com"),
    )
    prefix = keys.get("prefix", "scratch/imagenet")
    s3_bucket = s3_client.Bucket(keys.get("bucket_name", "iarai-playground"))
    return s3_bucket, prefix


# TODO  extract index file operations to super class and use common format for scratch and s3?
class S3Dataset(IndexedDataset):
    # TODO bucket_name or filename with s3_credentials? Pass in index file as init arg?
    def __init__(self, mode: str, bucket_name: str, limit: int = None) -> None:
        self.credentials_json = "s3_credentials.json"

        self.limit = limit
        # TODO magic string constants -> prefix to s3 configuration file?
        self.mode = "scratch/imagenet/" + mode
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])
        self.bucket_name = bucket_name
        self.rng = None
        self.load_index()
        self.len = len(self.image_paths)
        self.s3_bucket = None

    def lazy_init(self):
        """N.B.

        When using multiprocessing, instantiate the random generator in
        the spawned process (with the pid as seed) and instantiate boto3
        dynamically (as it is not picklable).
        """
        if self.s3_bucket is not None:
            return
        s3_bucket, prefix = get_s3_bucket(self.credentials_json)
        self.s3_bucket = s3_bucket
        self.rng = RandomGenerator(seed=os.getpid())

    # TODO should this not be part of dataloader?
    @overrides
    def get_random_item(self) -> Image:
        self.lazy_init()
        rn = self.rng.get_int(0, self.__len__())
        return self.__getitem__(rn)

    @stopwatch("(5)-get_item")
    def __getitem__(self, index: int) -> Image:
        self.lazy_init()
        b = BytesIO()
        self.s3_bucket.download_fileobj(self.image_paths[index], b)
        image = Image.open(b)
        return self.transform(image)

    def __len__(self):
        return self.len

    @staticmethod
    def index_all(credentials_json="s3_credentials.json", index_file: Optional[str] = None, file_ending="JPEG") -> None:
        s3_bucket, prefix = get_s3_bucket(credentials_json=credentials_json)
        data = [str(o) for o in s3_bucket.objects.filter(Prefix=prefix).all() if o.key.endswith(file_ending)]
        if index_file is not None:
            with open(index_file, "w") as f:
                json.dump(data, f)

    def load_index(self) -> None:
        # TODO magic string constants
        with open("index-s3.json", "r") as file:
            self.image_paths = [re.match(".*key='([^']+)", s).group(1) for s in json.load(file)]
