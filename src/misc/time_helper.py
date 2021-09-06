import json
import logging
import time
from functools import wraps
from timeit import default_timer as timer


def stopwatch(trace_name):
    def time_profiler(method):
        @wraps(method)
        def time_profile(*args, **kw):

            # https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
            ts = timer()
            ts_proc = time.process_time()
            result = method(*args, **kw)
            te = timer()
            te_proc = time.process_time()

            data = {
                "trace_name": trace_name,
                "function_name": method.__name__,
                "ms": (te - ts),
                "timestamp": time.time(),
                "process_time": (te_proc - ts_proc),
            }
            logging.getLogger("stopwatch").info(json.dumps(data))
            return result

        return time_profile

    return time_profiler
