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


python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output -a s3 --num_get_random_item 2000 --pool_size 20
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output -a s3 --num_get_random_item 2000 --pool_size 5
python -m benchmark.benchmark_dataset --output_base_folder ~/workspaces/storage-benchmarking/benchmark_output -a s3 --num_get_random_item 2000 --pool_size 0

python -c 'from dataset.scratch_dataset import ScratchDataset; from pathlib import Path; ScratchDataset.index_all(Path("/scratch/imagenet/val"), "index-scratch-val.json")'
python benchmark/benchmark_dataset.py -a scratch



python benchmark/benchmark_dataloader.py



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

Download index with: `aws s3 cp s3://iarai-playground/scratch/imagenet/index-s3-val.json .  --endpoint-url http://s3.amazonaws.com`

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

## Cite
Please cite this repo along with our [technical report](https://arxiv.org/abs/2211.04908):
```
@misc{svogor2022profiling,
      title={Profiling and Improving the PyTorch Dataloader for high-latency Storage: A Technical Report}, 
      author={Ivan Svogor and Christian Eichenberger and Markus Spanring and Moritz Neun and Michael Kopp},
      year={2022},
      eprint={2211.04908},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
