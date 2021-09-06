import importlib
import logging
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import h5py
import numpy as np
import torch
from dataset.s3_dataset import S3Dataset
from dataset.s3_file import S3File
from misc.time_helper import stopwatch

# source: https://github.com/iarai/NeurIPS2021-traffic4cast/blob/master/data/dataset/dataset.py

MAX_TEST_SLOT_INDEX = 240  # since a test goes over 2 hours, the latest possibility is 10p.m.


# However, `22*12 > 256 = 2^8` and so does not fit into uint8. Therefore, we (somewhat arbitrarily)
# chose to start the last test slot at 8-10p.m.


def load_h5_file(file_path: Union[str, Path], sl: Optional[slice] = None, to_torch: bool = False) -> np.ndarray:
    """Given a file path to an h5 file assumed to house a tensor, load that
    tensor into memory and return a pointer.
    Parameters
    ----------
    file_path: str
        h5 file to load
    sl: Optional[slice]
        slice to load (data is written in chunks for faster access to rows).
    """
    # load
    with h5py.File(str(file_path) if isinstance(file_path, Path) else file_path, "r") as fr:
        data = fr.get("array")
        if sl is not None:
            data = np.array(data[sl])
        else:
            data = np.array(data)
        if to_torch:
            data = torch.from_numpy(data)
            data = data.to(dtype=torch.float)
        return data


def prepare_test(
    data: np.ndarray, offset=0, to_torch: bool = False
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """Extracts an hour of test data for one hour and ground truth prediction
    5,10,15,30,45 and 60 minutes into the future.
    Parameters
    ----------
    data
        tensor of shape (24+, 495, 436, 8) of  type uint8
    offset
    to_torch:bool
        convert to torch float tensor.
    Returns
    -------
        test_data
            tensor of shape (12, 495, 436, 8) of  type uint8
        ground_truth_prediction
            tensor of shape (6, 495, 436, 8) of  type uint8
    """
    offsets = prepare_within_day_indices_for_ground_truth(offset)

    if isinstance(data, torch.Tensor):
        data = data.numpy()

    ub = offset + 12
    model_input = data[offset:ub]
    model_output = data[offsets]
    if to_torch:
        model_input = torch.from_numpy(model_input).float()
        model_output = torch.from_numpy(model_output).float()
    return model_input, model_output


def prepare_within_day_indices_for_ground_truth(offset: int) -> np.ndarray:
    """
    Parameters
    ----------
    offset: int
    Returns
    -------
        the 6 indices for the prediction horizon, i.e. offset+12, offset+13, ...., offset+23
    """
    return np.add([1, 2, 3, 6, 9, 12], 11 + offset)


class HDF5S3MODE(Enum):
    BOTO3_DOWNLOAD_FILE = 1
    RAWIO_WRAPPER_AROUND_BOTO3 = 2
    S3FS = 3


class T4CDataset(S3Dataset):
    def __init__(
        self,
        bucket_name: str,
        index_file: Path,
        # TODO should we support only relative paths instead of URLs?
        index_file_download_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        limit: Optional[int] = None,
        endpoint_url: Optional[str] = None,
        file_filter: str = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_npy: bool = False,
        mode: HDF5S3MODE = HDF5S3MODE.BOTO3_DOWNLOAD_FILE,
    ):
        """torch dataset from training data.
        Parameters
        ----------
        file_filter: str
            filter image_paths under `root_dir`, defaults to `"**/training/*ch8.h5`
        limit
            truncate dataset size
        transform
            transform applied to both the input and label
        """
        super().__init__(
            bucket_name=bucket_name,
            index_file=index_file,
            index_file_download_url=index_file_download_url,
            limit=limit,
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self.mode = mode

        self.file_filter = file_filter
        self.use_npy = use_npy
        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"
            if self.use_npy:
                self.file_filter = "**/training_npy/*.npy"
        self.transform = transform

    def _load_h5_file(self, fn, sl: Optional[slice]):
        if self.use_npy:
            return np.load(fn)
        else:
            return load_h5_file(fn, sl=sl)

    def __len__(self):
        size_240_slots_a_day = len(self.image_paths) * MAX_TEST_SLOT_INDEX
        if self.limit is not None:
            return min(size_240_slots_a_day, self.limit)
        return size_240_slots_a_day

    @stopwatch("(5)-get_item")
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx > self.__len__():
            raise IndexError("Index out of bounds")
        self.lazy_init()
        logging.info("get item %s", idx)

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX

        # / TODO #34 cleanup: analyse amount of data
        image_path = self.image_paths[file_idx]
        if self.mode == HDF5S3MODE.BOTO3_DOWNLOAD_FILE:
            temp = NamedTemporaryFile()
            file = temp.name
            self.s3_bucket.download_file(image_path, file)
        elif self.mode == HDF5S3MODE.RAWIO_WRAPPER_AROUND_BOTO3:
            boto3 = importlib.import_module("boto3")

            s3 = boto3.resource("s3")
            s3_object = s3.Object(bucket_name="iarai-playground", key=image_path)
            s3_file = S3File(s3_object)
            file = s3_file
        elif self.mode == HDF5S3MODE.S3FS:
            s3fs = importlib.import_module("s3fs")

            s3 = s3fs.S3FileSystem()
            # TODO #34 look at s3fs cache
            # s3.invalidate_cache() #noqa
            # s3.invalidate_region_cache()#noqa
            file = s3.open(f"s3://{self.bucket_name}/{image_path}", "rb")

        else:
            raise NotImplementedError()
        two_hours = self._load_h5_file(file, sl=slice(start_hour, start_hour + 12 * 2 + 1))
        # \

        input_data, output_data = prepare_test(two_hours)

        input_data = self._to_torch(input_data)
        output_data = self._to_torch(output_data)

        if self.transform is not None:
            input_data = self.transform(input_data)
            output_data = self.transform(output_data)

        return input_data, output_data

    def _to_torch(self, data):
        data = torch.from_numpy(data)
        data = data.to(dtype=torch.float)
        return data
