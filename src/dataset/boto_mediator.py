import importlib
import json
import re
from io import BytesIO

from PIL import Image


# loads S3 credentials
def get_s3_cred() -> json:
    __keys = None
    with open("s3_credentials.json", "r") as file:
        __keys = json.load(file)
    return __keys


# boto cannot be used in classes as is not pickable
boto_module = importlib.import_module("boto3")
keys = get_s3_cred()
s3_client = boto_module.resource(
    "s3",
    aws_access_key_id=keys["access_key"],
    aws_secret_access_key=keys["secret"],
    endpoint_url="http://s3.amazonaws.com",
)


def load_index() -> list:
    str_paths = None
    with open("index-s3.json", "r") as file:
        str_paths = json.load(file)
    image_paths = []
    # todo: find a cleverer way to do this (or another approach completely)
    for i in str_paths:
        [i := i.replace(j, "") for j in ",')"]
        i = re.findall(r'(\S+)=(".*?"|\S+)', i)
        image_paths.append(s3_client.ObjectSummary(i[0][1], i[1][1]))
    return image_paths


def save_index() -> None:
    str_paths = []
    [str_paths.append(str(i)) for i in image_paths]
    with open("index-s3.json", "w") as file:
        json.dump(str_paths, file)


def getitem(index: int = 1) -> Image:
    b = BytesIO()
    s3_client.Bucket("iarai-playground").download_fileobj(image_paths[index].key, b)
    image = Image.open(b)
    return image


image_paths = load_index()
