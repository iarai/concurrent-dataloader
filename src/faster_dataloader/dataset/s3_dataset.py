import argparse
import json
import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import tqdm
from overrides import overrides
from PIL import Image
from torchvision import transforms

from src.benchmarking.misc.random_generator import RandomGenerator
from src.benchmarking.misc.s3_helpers import download_file_from_s3_url
from src.benchmarking.misc.s3_helpers import get_s3_bucket
from src.benchmarking.misc.s3_helpers import upload_file_to_s3_url
from src.benchmarking.misc.time_helper import stopwatch
from src.faster_dataloader.dataset.indexed_dataset import IndexedDataset


# TODO  #32 extract index file operations to super class and use common format for scratch and s3?
class S3Dataset(IndexedDataset):
    def __init__(
        self,
        bucket_name: str,
        # TODO #32 make this optional, use temp file if not given
        index_file: Path,
        classes_file: Optional[Path] = None,
        # TODO should we support only relative paths instead of URLs?
        index_file_download_url: Optional[str] = None,
        classes_file_download_url: Optional[str] = None,
        limit: int = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        if index_file_download_url is not None and not index_file.exists():
            download_file_from_s3_url(
                s3_url=index_file_download_url,
                f=index_file,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                endpoint_url=endpoint_url,
            )

        if not classes_file.exists():
            download_file_from_s3_url(
                s3_url=classes_file_download_url,
                f=classes_file,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                endpoint_url=endpoint_url,
            )
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url
        self.index_file = index_file
        self.classes_file = classes_file
        self.limit = limit
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),])
        self.bucket_name = bucket_name
        self.rng = None
        self.classes = None
        self.use_cache = use_cache
        super().__init__(index_file=index_file, classes_file=classes_file)

        self.len = len(self.image_paths)
        self.s3_bucket = None
        self.s3_client = None

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
            use_cache=self.use_cache,
        )
        self.s3_bucket = s3_bucket
        self.rng = RandomGenerator(seed=os.getpid())

    # TODO #32 should this not be part of dataloader?
    @overrides
    def get_random_item(self) -> Image:
        self.lazy_init()
        rn = self.rng.get_int(0, self.__len__() - 1)
        return self.__getitem__(rn)

    # TODO we should do the do the @stopwatch instrumentalization only in the benchmarking part
    #  and keep this code clean from those aspects!
    # TODO #32 make this agnostic to Image or whatever
    @stopwatch(trace_name="(5)-get_item", trace_level=5, strip_result=True)
    def __getitem__(self, index: int, **kwargs) -> Tuple[Any, Any, Any]:
        """
        Check:
            https://pytorch.org/vision/stable/_modules/torchvision/datasets/imagenet.html
            https://github.com/pytorch/vision/blob/7947fc8fb38b1d3a2aca03f22a2e6a3caa63f2a0/torchvision/datasets/folder.py#L229
                - target is class_index of the target class
        """
        class_folder_name = self.image_paths[index].split("/")[3]
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
                    "validation image paths should start with 'ILSV'."
                )
        else:
            target = None

        self.lazy_init()
        b = BytesIO()
        self.s3_bucket.download_fileobj(self.image_paths[index], b)
        image = Image.open(b)
        if image.mode == "L":
            image = image.convert("RGB")
        return self.transform(image), target, b.getbuffer().nbytes

    def __len__(self):
        if self.limit is None:
            return self.len
        return min(self.len, self.limit)

    @staticmethod
    def index_all(
        bucket_name: str,
        index_file: Optional[Union[str, Path]] = None,
        file_ending="JPEG",
        prefix: str = "scratch/imagenet",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        index_file_upload_path: Optional[Union[str, Path]] = None,
    ) -> None:
        s3_bucket = get_s3_bucket(
            bucket_name=bucket_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
        )
        data = [str(o.key) for o in s3_bucket.objects.filter(Prefix=prefix).all() if o.key.endswith(file_ending)]
        logging.info("Indexed %s image_paths", len(data))
        if index_file is not None:
            logging.info("Writing to %s", index_file)
            with open(index_file, "w") as f:
                json.dump(data, f)

            if index_file_upload_path is not None:
                upload_file_to_s3_url(
                    s3_url=index_file_upload_path,
                    f=index_file,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    endpoint_url=endpoint_url,
                )

    @overrides
    def set_transform(self, transform: transforms) -> None:
        self.transform = transform


def s3_to_s3_copy(from_credentials: Path, to_credentials: Path, index_file_path: Path,) -> None:
    logging.info("Starting Copying ... Using S3")

    from_config = json.load(open(from_credentials))
    source_dataset = S3Dataset(index_file=Path(index_file_path), **from_config)
    source_dataset.load_index()
    source_dataset.lazy_init()

    logging.info(f"source_dataset has length {len(source_dataset)}")

    to_config = json.load(open(to_credentials))
    target_dataset = S3Dataset(index_file=Path(index_file_path), **to_config)
    target_dataset.lazy_init()

    logging.info(
        f"Uploading index file {index_file_path} "
        f"to bucket {to_config['bucket_name']} at endpoint url {to_config.get('endpoint_url', None)}"
    )
    target_dataset.s3_bucket.upload_file(str(index_file_path), str(index_file_path))
    logging.info(
        f"Uploading {len(source_dataset)} image_paths "
        f"from bucket {from_config['bucket_name']} at endpoint url {from_config.get('endpoint_url', None)}  "
        f"to bucket {to_config['bucket_name']} at endpoint url {to_config.get('endpoint_url', None)}"
    )
    for f in tqdm.tqdm(source_dataset.image_paths):
        temp = NamedTemporaryFile()
        source_dataset.s3_bucket.download_file(f, temp.name)
        target_dataset.s3_bucket.upload_file(temp.name, f)
    logging.info("End Copying ... Using S3")


def handle_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_credentials", type=Path, default=Path("s3_iarai_playground_imagenet.json"))
    parser.add_argument("--to_credentials", type=Path, default=Path("s3_credentials_eks.json"))
    parser.add_argument("--index_file_path", type=Path, default=Path("index-s3-val.json"))
    return parser


def main(*args):
    parser = handle_arguments()
    args = parser.parse_args(args)
    s3_to_s3_copy(**vars(args))


if __name__ == "__main__":
    main(*sys.argv[1:])
