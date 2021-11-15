import concurrent.futures
import json
import os
import random
from pathlib import Path
from timeit import default_timer as timer

import boto3
import numpy as np

# setup client and session

print("| implementation  | nodename | workers/actors |    payload |     elapsed | throughput|")
print("|--------|---------|-----------|--------------|-------------|------------------|")

for max_workers in [16, 32, 64, 128, 256]:
    sess = boto3.session.Session()
    client = sess.client("s3")
    with Path("index-s3-val.json").open("r") as f:
        files = json.load(f)

    random.shuffle(files)
    files = files[:20000]

    def download_from_s3(file_path):
        obj = client.get_object(Bucket="iarai-playground", Key=file_path)
        resp = obj["Body"].read()
        return len(resp)

    timer_start = timer()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_from_s3, files))
    time_end = timer()

    payload = np.sum(results)

    elapsed = time_end - timer_start
    payload_mb = payload / 10 ** 6
    rate_mb_sec = payload_mb * 8 / elapsed
    print(
        f"| boto3  | {os.uname().nodename} "
        f"| {max_workers:9.0f} | {payload_mb:10.2f}MB "
        f"| {elapsed:10.2f}s | {rate_mb_sec:10.2f}MBit/s |"
    )
