import importlib
from typing import Optional


# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
def get_s3_bucket(
    bucket_name: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
):
    boto3_module = importlib.import_module("boto3")
    s3_client = boto3_module.resource(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )
    s3_bucket = s3_client.Bucket(bucket_name)
    return s3_bucket
