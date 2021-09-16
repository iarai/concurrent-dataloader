r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
import concurrent
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

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
        self.items_flat = None

    def fetch_item(self, item, index):
        result = self.dataset.__getitem__(item)
        return {"tensor": self.collate_fn(result), "index": index, "item_id": item}

    def yield_item(self):
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

    @stopwatch(trace_name="(4)-threadedmapdataset-fetcher", trace_level=4)
    def yield_batch(self, items, items_flat, batch_sizes):
        self.items_flat = items_flat
        collected_batches = defaultdict(list)
        for r in self.yield_item():
            collected_batches[items[r["index"]]].append(r)
            for b in list(collected_batches):
                if len(collected_batches[b]) == batch_sizes[b]:
                    yield collected_batches[b], b
                    del collected_batches[b]
