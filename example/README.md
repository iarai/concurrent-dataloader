## Installation

To install the `fast-torch-downloader` use the setup file:

```
python3 setup.py install
```

or use the git repo with pip:

```
pip install git+http://url.todo
```

## Running the example

IMPORTANT: Make sure to create index, classes and credential files (described in the following section),

Simply run:

```commandline
python imagenet.py
```
or, to get detailed description of parameters, use:

```commandline
python imagenet.py --help
```

## Usage

For usage, consider `imagenet.py` project, which is essentially a [default implementation](https://github.com/pytorch/examples/blob/master/imagenet/main.py) provided by PyTorch. However, for clarity, some parts are removed.

To use the library, you need to make sure to create:

1) training file index (`imagenet-train-classes.json`),
2) classes file index (`imagenet-train-classes.json`),
3) S3-like credentials (depending on the kind of storage, this step may be unnecessary),
4) modify the data loading code (make sure to use process `fork`-ing).

### 1. Training file index

Normally, a `Dataset` instance is dealing with accessing raw files, which usually involves pointing to a directory,
listing all files, i.e. reading their names in a local variable, and later on return individual images using the `__getitem__` function.
However, with large datasets, listing folders can take some time, particularly during the coding phase, when still exploring
different parameters for the model. To overcome that issue, one can create a file index once, and then use it for subsequent
runs to access file directly. Here, we suggest a simple approach with a single list of paths, like:

```json
["scratch/imagenet/train/n01440764/n01440764_10026.JPEG",
"scratch/imagenet/train/n01440764/n01440764_10027.JPEG",
"scratch/imagenet/train/n01440764/n01440764_10029.JPEG",
"scratch/imagenet/train/n01440764/n01440764_10040.JPEG",
"scratch/imagenet/train/n01440764/n01440764_10042.JPEG",
"scratch/imagenet/train/n01440764/n01440764_10043.JPEG",
...]
```

Similarly, an index can be created for a validation dataset and a test dataset.

### 2. Classes file

Classes file is another `json` which associates each file in the file index, with its class. Imagenet uses 1000 classes, and each of
them has a textual name and an index. The association is made by the file name prefix, e.g. for `n01440764_10026.JPEG`, `n01440764` is the key
in the classes file, used to obtain its class. An example classes index structure is shown below:

```json
{
  "n02119789": {
    "id": 1,
    "name": "kit_fox"
  },
  "n02100735": {
    "id": 2,
    "name": "English_setter"
  },
  "n02110185": {
    "id": 3,
    "name": "Siberian_husky"
  },
  "n02096294": {
    "id": 4,
    "name": "Australian_terrier"
  },
  ...
}
```

Similarly, a validation classes file can be created with the following structure:

```json
{
"0": 490,
"1": 361,
"2": 171,
"3": 822,
...
```
where the key is the index of the image, and the value is its class.

`S3_Dataset` (`s3_dataset.py`) can also create an index file, and upload it to S3-like storage.

TODO: write how

### 3) S3-like credentials

A simple `json` file with the following structure:

```json
{
    "access_key": "...",
    "secret": "...I"
}
```

Also, this file can be extended to point to the other files used here, that might be downloaded from a remote location (e.g. prebuilt indexes):

```json
{
  "aws_access_key_id": "...",
  "aws_secret_access_key": "...",
  "bucket_name": "my_bucket",
  "train_classes_file_download_url": "s3://path/to/my_stuff/imagenet-train-classes.json",
  "val_classes_file_download_url": "s3://path/to/my_stuff/imagenet-val-classes.json",
  "val_index_file_download_url": "s3://path/to/my_stuff/index-s3-val.json",
  "train_index_file_download_url": "s3://path/to/my_stuff/index-s3-train.json",
  "test_index_file_download_url": "s3://path/to/my_stuff/index-s3-test.json"
}

```

When using the default `S3_Dataset` This file is used only if the `~/.aws` credentials are not present.

### 4) Modify the data loading code

#### 1. Import the library

```python
from benchmarking.misc.init_benchmarking import get_dataset
from concurrent_dataloader.dataloader_mod.dataloader import DataLoader as DataLoaderParallel
from concurrent_dataloader.dataloader_mod.worker import _worker_loop as _worker_loop_parallel
```

#### 2. Set the process creation method (at any setup point, before creating Dataset instances)

```python
    torch.multiprocessing.set_start_method("fork")
```

#### 3. Create datasets

```python
    # get the current folder
    base_folder = os.path.dirname(__file__)
    # get the path to the credetials files
    s3_credential_file = os.path.join(base_folder, "path/to/json/files/s3_credentials.json")
    # get index file names (in this case, shown only for the training dataset)
    train_dataset_index = f"path/to/json/files/index-{args.dataset}-train.json"

    # create a training dataset
    train_dataset = get_dataset(
        args.dataset,               # S3 or scratch (local storage)
        dataset_type="train",       # train or val(idation)
        limit=args.dataset_limit,   # max number of loaded items, dataset size
        use_cache=args.use_cache,   # use caching?
        index_file=Path(os.path.join(base_folder, train_dataset_index)),    # path to index file
        classes_file=Path(os.path.join(base_folder, "path/to/json/files/imagenet-train-classes.json")),
        s3_credential_file=s3_credential_file,  # credential file
    )
```

#### 4. Create dataloaders

```python
    # set transform (if data needs any)
    train_dataset.set_transform(transform)

    # Create the dataloader
    train_loader = DataLoaderParallel(
        dataset=train_dataset,                              # standard parameters
        batch_size=args.batch_size,                         # ...
        num_workers=args.num_workers,
        shuffle=(train_sampler is None),
        prefetch_factor=args.prefetch_factor,
        num_fetch_workers=args.num_fetch_workers,           # parallel threads used to load data
        fetch_impl=args.fetch_impl,                         # threads or asyncio
        batch_pool=args.batch_pool,                         # only for threaded implementation (pool of pre-loaded batches)
        pin_memory=True if args.pin_memory == 1 else False, # if using fork, it must be 0
    )
```

#### 5. Override the default Torch worker loop

Usually before using torch, or just after the previous code, point the standard `_worker_loop` to the modified one:

```python
    torch.utils.data._utils.worker._worker_loop = _worker_loop_parallel
```

**DONE!**

## Running in Colab

Running in Google Colab is easy. First of all, make sure to install all the requirements defined in the `environment.yaml`.
This can be done easily by copying the `pip` section in a separate folder, e.g. `requirements.txt`, and installing it:

```txt
!pip install -r requirements.txt
```

Furthermore, make sure that the Runtime accelerator is selected (Runtime > Change runtime type > Hardware accelerator: GPU):
To make sure that the GPU resources are properly assigned, simply run:

```commandline
import torch
device_index = 0
print(f"Running experiments on: {torch.cuda.get_device_name(device_index)}")
```
Make sure that `.json` index, credential and class files exist, and then, simply run:

```
!python imagenet.py --dataset-limit 3000
```
