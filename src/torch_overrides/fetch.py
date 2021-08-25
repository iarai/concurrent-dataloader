r""""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch
data from an iterable-style or map-style dataset. This logic is shared in both
single- and multi-processing data loading.
"""
import asyncio
import concurrent.futures
import functools

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
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_ThreadedMapDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.executor = concurrent.futures.ThreadPoolExecutor()

    def async_exec(self, f):
        async def aio_wrapper(**kwargs):
            f_bound = functools.partial(f, **kwargs)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, f_bound)

        return aio_wrapper

    def _fetch(self, possibly_batched_index):
        if self.auto_collation:
            data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)

    @stopwatch("(4)-threadedmapdataset-fetcher")
    def fetch(self, possibly_batched_index):
        self.async_exec(self._fetch(possibly_batched_index))