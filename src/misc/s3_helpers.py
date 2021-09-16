import importlib
from pathlib import Path
from typing import Optional
from typing import Union
from urllib.parse import urlparse

# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html


def get_s3_bucket(
    bucket_name: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    max_pool_connections: Optional[int] = 20,
):
    boto3_module = importlib.import_module("boto3")
    botocore_module = importlib.import_module("botocore")
    # https://github.com/boto/boto3/issues/801#issuecomment-358195444
    s3_client = None
    while not s3_client:
        try:
            s3_client = boto3_module.resource(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                endpoint_url=endpoint_url,
                config=botocore_module.client.Config(max_pool_connections=max_pool_connections),
            )
        except BaseException:
            s3_client = None

    s3_bucket = s3_client.Bucket(bucket_name)
    return s3_bucket


def upload_file_to_s3_url(
    f: Union[str, Path],
    s3_url: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
):
    s3_loc = urlparse(s3_url, allow_fragments=False)
    s3_bucket = get_s3_bucket(
        bucket_name=s3_loc.netloc,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )
    s3_bucket.upload_file(str(f), s3_loc.path)


def download_file_from_s3_url(
    f: Union[str, Path],
    s3_url: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
):
    s3_loc = urlparse(s3_url, allow_fragments=False)
    s3_bucket = get_s3_bucket(
        bucket_name=s3_loc.netloc,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )
    s3_bucket.download_file(s3_loc.path, str(f))
