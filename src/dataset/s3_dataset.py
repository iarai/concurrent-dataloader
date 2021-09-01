import json
import logging
import os
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional
from urllib.parse import urlparse

import tqdm
from PIL import Image
from overrides import overrides
from torchvision import transforms

from dataset.indexed_dataset import IndexedDataset
from misc.random_generator import RandomGenerator
from misc.s3_helpers import get_s3_bucket
from misc.time_helper import stopwatch


# TODO  #32 extract index file operations to super class and use common format for scratch and s3?
class S3Dataset(IndexedDataset):
    def __init__(
            self,
            bucket_name: str,
            index_file: Path,
            index_file_download_url: Optional[str] = None,
            limit: int = None,
            aws_access_key_id: Optional[str] = None,
            aws_secret_access_key: Optional[str] = None,
            endpoint_url: Optional[str] = None,
    ) -> None:

        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url
        self.index_file = index_file
        self.limit = limit
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), ])
        self.bucket_name = bucket_name
        self.rng = None

        if index_file_download_url is not None and not index_file.exists():
            s3_loc = urlparse(index_file_download_url, allow_fragments=False)
            s3_bucket = get_s3_bucket(
                bucket_name=bucket_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                endpoint_url=endpoint_url,
            )
            s3_bucket.download_file(s3_loc.path, str(index_file))
            del s3_bucket
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

        s3_bucket = get_s3_bucket(
            bucket_name=self.bucket_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
        )
        self.s3_bucket = s3_bucket
        self.rng = RandomGenerator(seed=os.getpid())

    # TODO #32 should this not be part of dataloader?
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
        if self.limit is None:
            return self.len
        return min(self.len, self.limit)

    @staticmethod
    def index_all(
            credentials_json="s3_dataset_configuration.json",
            index_file: Optional[str] = None,
            file_ending="JPEG",
            prefix: str = "scratch/imagenet",
    ) -> None:
        s3_bucket = get_s3_bucket(credentials_json=credentials_json)
        data = [str(o.key) for o in s3_bucket.objects.filter(Prefix=prefix).all() if o.key.endswith(file_ending)]
        if index_file is not None:
            with open(index_file, "w") as f:
                json.dump(data, f)

    def load_index(self) -> None:

        with open(self.index_file, "r") as file:
            self.image_paths = json.load(file)


def s3_to_s3_copy(from_credentials="s3_dataset_configuration.json", to_credentials="s3_dataset_configuration_temp_copy.json",
                  index_file_path="index-s3-val.json") -> None:
    logging.info("Starting Copying ... Using S3")

    from_config = json.load(open(from_credentials))
    source_dataset = S3Dataset(index_file=Path(index_file_path), **from_config)
    source_dataset.load_index()
    source_dataset.lazy_init()

    logging.info(f"source_dataset has length {len(source_dataset)}")

    to_config = json.load(open(to_credentials))
    target_dataset = S3Dataset(index_file=Path(index_file_path), **to_config)
    target_dataset.lazy_init()

    logging.info(f"Uploading index file {index_file_path} to bucket {to_config['bucket_name']} at endpoint url {to_config.get('endpoint_url', None)}")
    target_dataset.s3_bucket.upload_file(index_file_path, index_file_path)
    logging.info(
        f"Uploading {len(source_dataset)} files from bucket {from_config['bucket_name']} at endpoint url {from_config.get('endpoint_url', None)}  to bucket {to_config['bucket_name']} at endpoint url {to_config.get('endpoint_url', None)}")
    for f in tqdm.tqdm(source_dataset.image_paths):
        temp = NamedTemporaryFile()
        source_dataset.s3_bucket.download_file(f, temp.name)
        target_dataset.s3_bucket.upload_file(temp.name, f)
    logging.info("End Copying ... Using S3")
