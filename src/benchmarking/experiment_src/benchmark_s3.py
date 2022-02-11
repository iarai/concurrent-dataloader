from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import boto3
import time
from io import BytesIO
from pathlib import Path
import json 
from typing import Optional
from random import Random
import os
import pandas as pd 
from matplotlib import image 
import importlib

bucket_name = "iarai-playground"
aws_access_key_id = "5TUO0NL0QYOVEW7AZZLT"
aws_secret_access_key = "B2M1wP79AjZselDkTYZHYFppaemhiIiSesO0Luov"
endpoint_url = "http://10.0.2.1:80"
index_file = Path("../credentials_and_indexes/index-ceph-train.json")
botocore_module = importlib.import_module("botocore")
# local_path = "/iarai/home/ivan.svogor/temp/imagenet"
max_pool_connections = 400

class RandomGenerator:
    def __init__(self, seed: Optional[int] = 42):
        self.rng = Random()
        self.rng.seed(seed)

    def get_int(self, a, b):
        return self.rng.randint(a, b)

def download_one_file(bucket: str, client: boto3.client, s3_file: str):
    b = BytesIO()
    t_start = time.time()
    client.download_fileobj(bucket_name, s3_file, b)
    t_end = time.time()
    result = b.getbuffer().nbytes, t_end - t_start
    return result


def main():
    df = pd.DataFrame()

    with index_file.open("r") as file:
        image_paths = json.load(file)

    file_num = 500
    rng = RandomGenerator(seed=os.getpid())
    
    # Creating only one session and one client
    session = boto3.Session()
    client = session.client("s3", 
                             aws_access_key_id=aws_access_key_id, 
                             aws_secret_access_key=aws_secret_access_key, 
                             endpoint_url="http://10.0.2.1:80", 
                             config=botocore_module.client.Config(max_pool_connections=max_pool_connections),
                             )

    # The client is shared between threads
    image_paths_to_dl = [image_paths[rng.get_int(0, len(image_paths))] for _ in range(file_num)]
    # print(image_paths_to_dl)

    func = partial(download_one_file, bucket_name, client)
    for i in [20, 60, 100, 300]:
        t_start = time.time()
        dl_details = []
        with ThreadPoolExecutor(max_workers=i) as executor:
            # for _ in range(file_num):
            #     # futures = { executor.submit(func, image_paths[rng.get_int(0, len(image_paths))]) }
            #     for _, future in enumerate(as_completed(futures)):
            #         try:
            #             dl_details.append(future.result())
            #         except Exception as ex:
            #             print(f"Exception {ex}")
                # futures = { executor.submit(func, image_paths[rng.get_int(0, len(image_paths))]) }
            futures = {
                executor.submit(func, file_to_download): file_to_download for file_to_download in image_paths_to_dl
            }
            for _, future in enumerate(as_completed(futures)):
                try:
                    dl_details.append(future.result())
                except Exception as ex:
                    print(f"Exception {ex}")
        t_end = time.time()

        total_bytes = 0
        total_time = 0 
        [total_bytes := total_bytes + i for i, _ in dl_details]
        [total_time := total_time + j for _, j in dl_details]

        total_time_all = t_end - t_start
        result = {
            "images": file_num, 
            "total_size_bytes": total_bytes,
            "total_time_individual_sum_sec": total_time,
            "total_time_all_sec": total_time_all,
            "throughput_t_ind_mbs": ( total_bytes / 1e6 ) / total_time_all, 
            "throughput_t_ind_mbits": (total_bytes * 8 / 1e6 ) / total_time_all, 
            "throughput_t_all_mbs": (total_bytes / 1e6) / total_time,
            "throughput_t_all_mbits": (total_bytes * 8 / 1e6) / total_time,
            "throughput_img_s_t_ind": file_num / total_time,
            "throughput_img_s_t_all": file_num / total_time_all,
        }
        df = df.append(pd.Series(data=result), ignore_index=True)
        print(df)

if __name__ == "__main__":
    main()