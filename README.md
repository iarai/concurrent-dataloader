## Storage Benchmarking

Work in progress...

Use as:

`python3 main.py -a s3`

or to see help:

`python3 main.py -help`

## Activate conda environment @lnx-slim-2

 1) `conda-bash`
 2) `conda env create -f environment.yml`
 3) `conda activate storage-benchmarking`

If using `tu-` server with A100:

`conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

## Pre-commit checkup

 `pre-commit run --all`

## Run experiments

Use the provide bash script:

```buildoutcfg
$ cd src

$ ./run.sh
```
By default, the script uses `s3` storage, however, it can also test `scratch` storage. To change this change `DATASOURCE='s3'` to `scratch`.

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
