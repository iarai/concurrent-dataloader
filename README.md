## Storage Benchmarking

## Activate conda environment on computes

```
conda-bash
conda env update -f environment.yml
conda activate storage-benchmarking
```

If using `tu-` server with A100:

`conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

## Run experiments
Work in progress...

Use as:

```
cd src
export PYTHONPATH=$PWD
python benchmark/benchmark_tensor_loading.py -a random_gpu
python benchmark/benchmark_tensor_loading.py -a random_to_gpu
python benchmark/benchmark_tensor_loading.py -a single_image
python benchmark/benchmark_tensor_loading.py -a random_image
python benchmark/benchmark_tensor_loading.py -a mp


vi s3_iarai_playground_imagenet.json # see below
python benchmark/benchmark_dataset.py -a s3
python -c 'from dataset.scratch_dataset import ScratchDataset; from pathlib import Path; ScratchDataset.index_all(Path("/scratch/imagenet/val"), "index-scratch-val.json")'
python benchmark/benchmark_dataset.py -a scratch



python benchmark/benchmark_dataloader.py



```


## sync tu -> gluster
```
rsync -Wuva ~/workspaces/storage-benchmarking/src/benchmark_output/ christian.eichenberger@lnx-slim-1.lan.iarai.ac.at:/iarai/work/logs/storage_benchmarking/
```









## `IndexedDataset`

The dataset classes inheriting from `IndexedDataset` are supposed to have json index file with the file paths in them and the dataset.
For instance, `index-s3-val.json` for the validation set of imagenet looks like this in the `iarai-playground` bucket:
```json
[
   "scratch/imagenet/val/ILSVRC2012_val_00000001.JPEG",
   "scratch/imagenet/val/ILSVRC2012_val_00000002.JPEG",
   "scratch/imagenet/val/ILSVRC2012_val_00000003.JPEG",
   "scratch/imagenet/val/ILSVRC2012_val_00000004.JPEG",
   ...
]
```
Similarly, for a local `IndexedDataset`, the bucket corresponds to a local root folder and the paths would correspond to the relative paths under that root folder.

Apart from the index file, we do not want the dataset interface to depend on any particular file format for the specification of the S3 endpoint.
(We might want to get rid of that dependency as well?)
Hence, we can instantiate like this, passing the file location to download the index file to and reading other parameters (aws credentials, bucket name and index file download url) from a json file:
```
s3_dataset_configuration = json.load(open("s3_dataset_configuration.json"))
s3_dataset = S3Dataset(index_file=Path("index-s3-val.json"), **s3_dataset_configuration)
```
Here, `s3_dataset_configuration.json` looks like this:
```
{
  "aws_access_key_id": "XXX",
  "aws_secret_access_key": "XXX",
  "bucket_name": "iarai-playground",
  "index_file_download_url": "s3://iarai-playground/scratch/imagenet/index-s3-val.json"
}
```
Notice that if we do not pass credentials directly, `boto3` will look them up in the standard locations according to https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html.
However, this files give us the flexibility to use custom s3 locations flexibly with relying on a central aws configuration and make these configurations shareable.

### Creating the index file

Create a local index for data in a s3 bucket:
```
s3_dataset = S3Dataset.index_all(
   index_file=Path("index-t4c.json"),
   file_ending="_8ch.h5",
   prefix="scratch/neun_t4c/raw",
   bucket_name="iarai-playground",
   # TODO use relative path within the bucket?
   index_file_upload_path="s3://scratch/neun_t4c/t4c21-index.json"
)
```

## Pre-commit checkup

`pre-commit run --all`
