r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from typing import List

import torch
from misc.time_helper import stopwatch
from torch.utils.data import Dataset


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

    @stopwatch(trace_name="(4)-mapdataset-fetcher", trace_level=4)
    def fetch(self, possibly_batched_index):
        # print("Overriden... {possibly_batched_index}")
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


class _ThreadedMapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(
        self, dataset: Dataset, auto_collation: bool, collate_fn: Callable, drop_last: bool, num_fetch_workers: int = 1
    ):
        super(_ThreadedMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        # Initialize a thread pool
        self.thread_pool_size = num_fetch_workers
        self._executor = ThreadPoolExecutor(num_fetch_workers)
        # Create a new async event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def worker(self, worker_id: int, task_queue: asyncio.Queue, result_queue: asyncio.Queue) -> None:
        while not task_queue.empty():
            index = await task_queue.get()
            try:
                result = await self.loop.run_in_executor(self._executor, self.dataset.__getitem__, index)
                result = self.collate_fn(result)
                result_queue.put_nowait((index, result))
            except Exception as e:
                print(f"Exception in fetch worker {worker_id}: {str(e)}")
            finally:
                task_queue.task_done()

    async def initiate_fetch_tasks(self, batch_indices: List[int]) -> List[torch.Tensor]:
        """Creates a list of tasks and initiates their execution using a thread
        pool.

        Arguments:
           batch_indices -- a list of integers which represent the index of an
           item that should be fetched by the dataset
        Returns:
           a list of tensor objects sorted in accordance to the batch_indices order
        """
        # Create input and output queue, task_, result_ to store the tasks and their results (respectively)
        task_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        # load indexes into a queue
        for index in batch_indices:
            task_queue.put_nowait(index)

        # create tasks and run
        tasks = [asyncio.create_task(self.worker(i, task_queue, result_queue)) for i in range(self.thread_pool_size)]

        # await for results
        await asyncio.gather(*tasks, return_exceptions=True)

        # read the result queue
        result_list = []
        while not result_queue.empty():
            result_list.append(result_queue.get_nowait())

        # sort wrt index
        return result_list.sort(key=lambda v: v[0])

    @stopwatch(trace_name="(4)-threadedmapdataset-fetcher", trace_level=4)
    def fetch(self, batch_indices: List[int]) -> List[torch.Tensor]:
        """Entrypoint function to async execution. It calls the async function
        initiate_fetch_tasks that uses batch indices to create a list of tasks
        that are performed asynchronously.

        - This fetch function cannot be async itself, otherwise, it would need to be awaited by it's caller.

        Arguments:
           batch_indices -- a list of integers which represent the index of an
           item that should be fetched by the dataset
        Returns:
           a list of tensor objects (fetched items, by the dataset.__getitem__
           and with the predefined transformations applied)
        """
        # create a future that waits for all tasks to complete
        result = self.loop.run_until_complete(self.initiate_fetch_tasks(batch_indices))
        return result
