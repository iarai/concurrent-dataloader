import importlib
import json
import re
from io import BytesIO

from PIL import Image


def get_s3_cred() -> None:
    __keys = None
    with open("s3_credentials.json", "r") as file:
        __keys = json.load(file)
    return __keys


def load_index():
    str_paths = None
    with open("index-s3.json", "r") as file:
        str_paths = json.load(file)
    image_paths = []
    # todo: find a cleverer way to do this (or another approach completley)
    for i in str_paths:
        [i := i.replace(j, "") for j in ",')"]
        i = re.findall(r'(\S+)=(".*?"|\S+)', i)
        image_paths.append(s3_client.ObjectSummary(i[0][1], i[1][1]))
    return image_paths


keys = get_s3_cred()
boto_module = importlib.import_module("boto3")

s3_client = boto_module.resource("s3", aws_access_key_id=keys["access_key"],
                                 aws_secret_access_key=keys["secret"],
                                 endpoint_url="http://s3.amazonaws.com")

def get_client():
    return s3_client


def li():
    str_paths = None
    with open("index-s3.json", "r") as file:
        str_paths = json.load(file)
    image_paths = []
    for i in str_paths:
        [i := i.replace(j, "") for j in ",')"]
        i = re.findall(r'(\S+)=(".*?"|\S+)', i)
        image_paths.append(s3_client.ObjectSummary(i[0][1], i[1][1]))
    return image_paths

image_paths = li()

def getitem(index: int = 1) -> Image:
    # print(f"imgs {len(image_paths)}")
    b = BytesIO()
    # print(f"Index: {index}")
    s3_client.Bucket("iarai-playground").download_fileobj(image_paths[index].key, b)
    image = Image.open(b)
    return image
    # return self.transform(image)
