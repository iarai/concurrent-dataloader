r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
import concurrent.futures
import asyncio
import os
from misc.time_helper import stopwatch
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor


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


# class _ThreadedMapDatasetFetcher(_BaseDatasetFetcher):
#     def __init__(self, dataset, auto_collation, collate_fn, drop_last):
#         super(_ThreadedMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
#
#     def _fetch_item(self, name, index):
#         print(f"Fetching {index} from {name} + {os.getpid()}")
#         data = self.dataset.__getitem__(index, worker_name=name)
#         return self.collate_fn(data)
#
#     @stopwatch("(4)-threadedmapdataset-fetcher")
#     def fetch(self, batch_indexes):
#         # create a thread pool executor
#         executor = ThreadPoolExecutor(max_workers=3)
#         # prepare the tasks
#         futures = []
#         for index in batch_indexes:
#             futures.append(executor.submit(self._fetch_item, index=index, name="name"))
#         # wait till all tasks are completed
#         futures, _ = concurrent.futures.wait(futures)
#         # return the result
#         result = [f.result() for f in futures]
#         return result

# class _ThreadedMapDatasetFetcher(_BaseDatasetFetcher):
#     def __init__(self, dataset, auto_collation, collate_fn, drop_last):
#         super(_ThreadedMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
#         self.queue = asyncio.Queue()
#         self.pool_size = 4
#         self.result = []
#
#     async def _fetch_item(self, name, index):
#         print(f"Worker {name} downloading {index} ... {os.getpid()}")
#         data = self.dataset.__getitem__(index, worker_name=name)
#         if data is not None:
#             return self.collate_fn(data)
#         else:
#             return None
#
#     async def worker(self, name, q):
#         while True:
#             index = await q.get()
#             try:
#                 data = await self._fetch_item(name=name, index=index)
#                 self.result.append(data[0])
#             finally:
#                 q.task_done()
#
#     async def async_exec(self, batch_indexes):
#         # Create a queue that we will use to store our "workload".
#         queue = asyncio.Queue()
#         # load indexes into a queue
#         for index in batch_indexes:
#             queue.put_nowait(index)
#
#         # create tasks and run
#         tasks = [asyncio.create_task(self.worker(f'worker-{i}-{os.getpid()}', queue)) for i in range(5)]
#
#         await queue.join()
#
#         # Cancel our worker tasks.
#         for task in tasks:
#             task.cancel()
#         # Wait until all worker tasks are cancelled.
#         # await asyncio.gather(*tasks, return_exceptions=True)
#         return self.result
#
#     @stopwatch("(4)-threadedmapdataset-fetcher")
#     def fetch(self, batch_indexes):
#         self.result.clear()
#         return asyncio.run(self.async_exec(batch_indexes))

_executor = ThreadPoolExecutor(3)
loop = asyncio.get_event_loop()
class _ThreadedMapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_ThreadedMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.pool_size = 4
        self.result = []

    def _fetch_item(self, name, index):
        print(f"Worker {name} downloading {index} ... {os.getpid()}")
        data = self.dataset.__getitem__(index, worker_name=name)
        return self.collate_fn(data)

    async def worker(self, name, q):
        while not q.empty():
            print("Here...")
            index = await q.get()
            try:
                result = await loop.run_in_executor(_executor, self._fetch_item, name, index)
                # result = await self._fetch_item(name=name, index=index)
                self.result.append(result[0])
            except Exception as e:
                print(f"Exception {str(e)}")
            finally:
                print(f"{name} done {q}")
                q.task_done()

    async def async_exec(self, batch_indexes):
        # Create a queue that we will use to store our "workload".
        queue = asyncio.Queue()
        # load indexes into a queue
        for index in batch_indexes:
            queue.put_nowait(index)

        # create tasks and run
        tasks = [asyncio.create_task(self.worker(f'worker-{i}-{os.getpid()}', queue)) for i in range(3)]
        await asyncio.gather(*tasks, return_exceptions=True)
        await queue.join()

        # Cancel our worker tasks.
        for task in tasks:
            task.cancel()
        # Wait until all worker tasks are cancelled.
        await asyncio.gather(*tasks, return_exceptions=True)
        return self.result

    @stopwatch("(4)-threadedmapdataset-fetcher")
    def fetch(self, batch_indexes):
        self.result.clear()
        # return asyncio.run(self.async_exec(batch_indexes))

        result = loop.run_until_complete(self.async_exec(batch_indexes))
        # loop.close()
        return result
        # return self.result