r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from typing import Type

import torch
from misc.time_helper import stopwatch


class _BaseDatasetFetcher(object):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError()


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_MapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)

    @stopwatch("(4)-mapdataset-fetcher")
    def fetch(self, possibly_batched_index):
        # print("Overriden... {possibly_batched_index}")
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


class _ThreadedMapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last, num_fetch_workers=1):
        super(_ThreadedMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.thread_pool_size = num_fetch_workers
        self._executor = ThreadPoolExecutor(self.thread_pool_size)
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            pass  # log here...
        finally:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        # print(f"Fetching with {num_fetch_workers}")

    def _fetch_item(self, worker_id: str, index: int) -> torch.Tensor:
        # print(f"Worker {name} downloading {index} ... {os.getpid()}")
        data = self.dataset.__getitem__(index, worker_name=worker_id)
        return self.collate_fn(data)

    async def worker(self, worker_id: str, task_queue: asyncio.Queue, result_queue: asyncio.Queue) -> None:
        while not task_queue.empty():
            index = await task_queue.get()
            try:
                result = await self.loop.run_in_executor(self._executor, self._fetch_item, worker_id, index)
                result_queue.put_nowait((index, result))
            except Exception as e:
                print(f"Exception in fetch worker {worker_id}: {str(e)}")
            finally:
                task_queue.task_done()

    async def async_exec(self, batch_indexes: List[int]) -> List[torch.Tensor]:
        # Create a queue that we will use to store our "workload".
        task_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        # load indexes into a queue
        for index in batch_indexes:
            task_queue.put_nowait(index)

        # create tasks and run
        tasks = [asyncio.create_task(self.worker(i, task_queue, result_queue)) for i in range(self.thread_pool_size)]

        # await for results
        await asyncio.gather(*tasks, return_exceptions=True)
        await task_queue.join()

        # Cancel worker tasks.
        for task in tasks:
            task.cancel()
        # Wait until all worker tasks are cancelled.
        await asyncio.gather(*tasks, return_exceptions=True)

        # read the result queue
        result_list = []
        while not result_queue.empty():
            result_list.append(result_queue.get_nowait())
        # sort wrt index
        return result_list.sort(key=lambda tup: tup[0])

    @stopwatch("(4)-threadedmapdataset-fetcher")
    def fetch(self, batch_indexes: List[int]) -> List[torch.Tensor]:
        result = self.loop.run_until_complete(self.async_exec(batch_indexes))
        return result
