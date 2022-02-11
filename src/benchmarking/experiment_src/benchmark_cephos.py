import importlib
from typing import Optional
from urllib.parse import urlparse
import os
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Any
from typing import Optional
from pathlib import Path
from benchmarking.misc.random_generator import RandomGenerator
from benchmarking.misc.s3_helpers import get_s3_bucket
from PIL import Image
import json
from random import Random
from typing import Optional
import time
from numpy import product, result_type
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import multiprocessing
import threading
import queue


def get_s3_bucket(
    bucket_name: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    max_pool_connections: Optional[int] = 500,
    max_retries: int = 100,
):

    boto3_module = importlib.import_module("boto3")
    botocore_module = importlib.import_module("botocore")
    # https://github.com/boto/boto3/issues/801#issuecomment-358195444
    s3_client = None
    retries = 0
    while not s3_client and retries < max_retries:
        retries += 1
        try:
            s3_client = boto3_module.resource(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                endpoint_url=endpoint_url,
                config=botocore_module.client.Config(max_pool_connections=max_pool_connections),
            )
        except Exception as ex:
            s3_client = None

    if retries >= 100:
        raise InterruptedError("Max retries for creating of s3_client reached!")

    s3_bucket = s3_client.Bucket(bucket_name)
    return s3_bucket


class RandomGenerator:
    def __init__(self, seed: Optional[int] = 42):
        self.rng = Random()
        self.rng.seed(seed)

    def get_int(self, a, b):
        return self.rng.randint(a, b)


class Downloader:
    def __init__(self, mode = "remote"):
        self.s3_bucket = None
        self.rng = None
        self.mode = mode


        self.bucket_name = "iarai-playground"
        self.local_path = "/iarai/home/ivan.svogor/temp/imagenet"
        
        # CEPH-OS

        if mode == "ceph-os" or mode == "local":
            self.aws_access_key_id = "5TUO0NL0QYOVEW7AZZLT"
            self.aws_secret_access_key = "B2M1wP79AjZselDkTYZHYFppaemhiIiSesO0Luov"
            self.endpoint_url = "http://10.0.2.1:80"
            self.index_file = Path("../credentials_and_indexes/index-ceph-train.json")
        elif mode == "s3":
        # AWS
            self.aws_access_key_id = "AKIA3WN63774XSV2B6TZ"
            self.aws_secret_access_key = "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"
            # self.endpoint_url = "http://s3.amazonaws.com"
            self.endpoint_url = "http://s3.eu-west-1.amazonaws.com"
            self.index_file = Path("../credentials_and_indexes/index-s3-train.json")

        with self.index_file.open("r") as file:
            self.image_paths = json.load(file)
    
    def lazy_init(self):
        if self.s3_bucket is not None:
            return

        s3_bucket = get_s3_bucket(
            bucket_name=self.bucket_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
        )
        self.s3_bucket = s3_bucket
        self.rng = RandomGenerator(seed=os.getpid())


    def download_many(self, num, pool_size):
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            __futures = { executor.submit(self.get_random_item) for _ in range(num) }
            for future in as_completed(__futures):
                try:
                    yield future.result()
                except Exception as exc:
                    print(f"Exception in fetcher: {str(exc)}")

    def get_random_item(self, c = None) -> Image:
        self.lazy_init()
        random_image_path = self.image_paths[self.rng.get_int(0, self.__len__() - 1)]
        # print(f"Downloading: {random_image_path}, {os.getpid(), time.time(), threading.current_thread().name}")
        t_start = time.time()
        image_size = self.download_item(random_image_path)
        t_end = time.time()
        result = image_size, t_end - t_start
        if type(c) == queue.Queue:
            c.put(result)
        else:
            return result

    def download_item(self, image_path):
        if self.mode != "local":
            b = BytesIO()
            self.s3_bucket.download_fileobj(image_path, b)
            _ = Image.open(b)
            if b.getbuffer().nbytes == 0:
                raise Exception("Downloaded object has size 0.")
            return b.getbuffer().nbytes
        elif self.mode == "local":
            img = Image.open(self.local_path + "/" + image_path)
            return len(img.fp.read())
        else:
            raise Exception("Unknown mode")

    def __len__(self):
        return len(self.image_paths)


def threaded(dl, file_num):
    result = []
    threads = []
    result_queue = queue.Queue()
    for _ in range(file_num):
        threads.append(threading.Thread(target=dl.get_random_item, args=(result_queue,)))
    [t.start() for t in threads]
    [t.join() for t in threads]
    while not result_queue.empty():
        result.append(result_queue.get())  
    return result

def mp_threaded(dl, file_num, mp_q):
    print(f"Download {file_num}")
    threads = []
    result_queue = queue.Queue()
    for _ in range(file_num):
        threads.append(threading.Thread(target=dl.get_random_item, args=(result_queue,)))
    [t.start() for t in threads]
    [t.join() for t in threads]
    while not result_queue.empty():
        r = result_queue.get()
        mp_q.put(r)

def mp_concurrent(dl, file_num, mp_q):
    for i, j in dl.download_many(file_num, 16):
        mp_q.put((i, j))

def main(dl_mode):
    dl = Downloader(mode = dl_mode)
    df = pd.DataFrame()
    modes = ["basic", "concurrent", "multiprocessing", "multithreading", "hybrid", "proc-thr"]
    mode = modes[5]
    pool_size = 5

    for file_num in [5000] * 5:
        print(f"Running with: {file_num}, concurrency mode: {mode}, storage: {dl_mode}")
        dl_details = []
        t_start = time.time()

        if mode == "basic":
            dl_details = [dl.get_random_item() for _ in range(file_num)]
        elif mode == "concurrent":
            for i, j in dl.download_many(file_num, pool_size):
                dl_details.append((i, j))
        elif mode == "multiprocessing":
            with multiprocessing.Pool(processes = pool_size) as pool:
                dl_details =  [pool.apply_async(dl.get_random_item, args=(_,)).get() for _ in range(file_num)]
        elif mode == "multithreading":
            dl_details = threaded(dl, file_num)
        elif mode == "hybrid":
            dl_details = []
            with multiprocessing.Pool(processes = pool_size) as pool:
                dl_details =  [pool.apply_async(threaded, args=(dl, file_num // pool_size,)).get() for _ in range(pool_size)]
            dl_details = [item for sublist in dl_details for item in sublist]
        elif mode == "proc-thr":
            queues = []
            processes = []
            dl_details = []
            for _ in range(pool_size):
                mp_q = multiprocessing.Queue()
                processes.append(multiprocessing.Process(target=mp_concurrent, args=(dl, file_num // pool_size, mp_q)))
                queues.append(mp_q)
            [p.start() for p in processes]
            [p.join() for p in processes]
            for q in queues:
                while not q.empty():
                    r = q.get()
                    dl_details.append(r)
        t_end = time.time()

        
        total_bytes = 0
        total_time = 0 
        
        [total_bytes := total_bytes + i for i, _ in dl_details]
        [total_time := total_time + j for _, j in dl_details]

        total_time_all = t_end - t_start
        mb_dif = 1024 * 1024 #1e6
        result = {
            "images": file_num, 
            "total_mb": total_bytes / mb_dif,
            "total_time_all_sec": total_time_all,
            "throughput_mbps": (total_bytes / mb_dif) / total_time_all,
            "throughput_mbitps": (total_bytes * 8 / mb_dif) / total_time_all,
            "throughput_imgps": file_num / total_time_all,
        }
        df = df.append(pd.Series(data=result), ignore_index=True)
        print(df)

if __name__ == "__main__":
    for mode in ["s3", "ceph-os", "local"]: 
        main(mode)