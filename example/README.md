## Installation

To install the `fast-torch-downloader` use the setup file:

```
python3 setup.py install
```

or use the git repo with pip:

```
pip install git+http://url.todo
```

## Usage

For usage, consider `imagenet.py` project, which is essentially a [default implementation](https://github.com/pytorch/examples/blob/master/imagenet/main.py) provided by PyTorch. However, for clarity, some parts are removed. 

To use the library, you need to make sure to create:

1) training file index 
2) classes file index
3) OPTIONAL: if using remote storage, proper S3 credentials
4) modify Dataloading code
5) make sure to use process forking 

### 1. Training file index

Training file index can be a `.json` file that is used in the Dataset. In this example, we are using `faster_downloader/dataset/S3_dataset.py` that uses the index file of Imagenet dataset in the following format:

```
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

With this, one can easily access items using the file index, i.e. `id` which saves a lot of loading time with large datasets. 
`S3_Dataset` can also create an index file, and upload it to S3-like storage.  

TODO: write how 

### 2. Classes file 