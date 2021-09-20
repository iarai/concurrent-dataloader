import importlib
import json
# setup client and session
import os
import random
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import ray
from ray.util import ActorPool

print("| implementation  | nodename | workers/actors |    payload |     elapsed | throughput|")
print("|--------|---------|-----------|--------------|-------------|------------------|")

@ray.remote
class AsyncActor:
    def __init__(self):
        boto3 = importlib.import_module("boto3")
        sess = boto3.session.Session()
        self.client = sess.client("s3")

    async def run_task(self, file_path):
        obj = self.client.get_object(Bucket="iarai-playground", Key=file_path)
        resp = obj["Body"].read()
        return len(resp)


for num_actors in [16, 32, 64, 128, 256]:
    with Path("src/index-s3-val.json").open("r") as f:
        files = json.load(f)

    random.shuffle(files)
    files = files[:20000]

    pool = ActorPool([AsyncActor.remote() for _ in range(num_actors)])
    timer_start = timer()
    results = list(pool.map(lambda a, v: a.run_task.remote(v), files))
    time_end = timer()

    payload = np.sum(results)

    elapsed = time_end - timer_start
    payload_mb = payload / 10 ** 6
    rate_mb_sec = payload_mb * 8 / elapsed
    print(
        f"| rayasyncactor  | {os.uname().nodename} | {num_actors:9.0f} | {payload_mb:10.2f}MB | {elapsed:10.2f}s | {rate_mb_sec:10.2f}MBit/s |")
    # print(f"max_workers{max_workers:9.f}")
    # print(f"payload: {payload_mb:10.2f}MB")
    # print(f"elapsed: {elapsed:10.2f}s")
    # print(f"rate:    {rate_mb_sec:10.2f}MBit/s")
