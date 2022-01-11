r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
# // Modified: added libraries for parallel downloads
import asyncio
import concurrent
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from typing import List

from benchmarking.misc.time_helper import stopwatch
from torch import Tensor
from torch.utils.data import Dataset

# \\


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

    # // Modified: added for logging
    @stopwatch(trace_name="(4)-mapdataset-fetcher", trace_level=4)
    # \\
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


# // Modified: added two new classes to parallelize data fetching -- using Asyncio
class _AsyncMapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(
        self, dataset: Dataset, auto_collation: bool, collate_fn: Callable, drop_last: bool, num_fetch_workers: int = 1
    ):
        super(_AsyncMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
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
                result_queue.put_nowait((index, result))
            except Exception as e:
                print(f"Exception in fetch worker {worker_id}: {str(e)}")
            finally:
                task_queue.task_done()

    async def initiate_fetch_tasks(self, batch_indices: List[int]) -> List[Tensor]:
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
        result_list.sort(key=lambda v: v[0])
        # collate the batch (index 0 are indexes, not necessary after sorting)
        if self.collate_fn is not None:
            return self.collate_fn(result_list)[1]
        return result_list

    @stopwatch(trace_name="(4)-asyncmapdataset-fetcher", trace_level=4)
    def fetch(self, batch_indices: List[int]) -> List[Tensor]:
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


# \\

# // Modified: added two new classes to parallelize data fetching -- using multiple threads
class _ThreadedMapDatasetFetcher(_BaseDatasetFetcher):
    def __init__(
        self, dataset: Dataset, auto_collation: bool, collate_fn: Callable, drop_last: bool, num_fetch_workers: int = 1
    ):
        super(_ThreadedMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        # Initialize a thread pool
        self.thread_pool_size = num_fetch_workers
        self.items_flat = None

    def fetch_item(self, item: int, index: int) -> dict:
        """Uses the dataset __getitem__ function to fetch a single data item.

        Arguments
            - item, data item id (not to be confused with index)
            - index, index of the element in the batch (used for sorting results)
        Returns:
            - a dictionary of
                - tensor, resulting item (not necessarily a tensor)
                - index, item's position in the batch
                - item_id, item identifier (=input argument item) for book keeping
        """
        result = self.dataset.__getitem__(item)
        return {"tensor": result, "index": index, "item_id": item}

    @stopwatch(trace_name="(4)-threadedmapdataset-fetcher", trace_level=4)
    def yield_item(self) -> dict:
        """Uses a ThreadPoolExecutor and creates a list of futures, i.e. tasks.
        Each task returns a single data item, and as the results come in, they
        are yielded.

        Returns:
            - a dictionary with tensor, data item id, and it's index in the batch
            (check return from the `fetch_item` function)
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_pool_size) as executor:
            futures = {
                executor.submit(self.fetch_item, item, index): item for item, index in enumerate(self.items_flat)
            }
            for future in concurrent.futures.as_completed(futures):
                data = futures[future]
                try:
                    data = future.result()
                    yield data
                except Exception as exc:
                    print(f"Exception in fetcher: {str(exc)}")

    # @stopwatch(trace_name="(4)-threadedmapdataset-fetcher", trace_level=4)
    def yield_batch(self, items, batch_sizes) -> dict:
        """Arguments.

            - items, indices of  all items in the batch
            - batch_sizes, size of each batch
        Returns
            - complete batch, as dictionary with items, batch indexes and item_ids
        """
        self.items_flat = list(items.keys())
        collected_batches = defaultdict(list)
        for r in self.yield_item():
            collected_batches[items[r["index"]]].append(r)
            for b in list(collected_batches):
                if len(collected_batches[b]) == batch_sizes[b]:
                    yield collected_batches[b], b
                    del collected_batches[b]


# \\
