import json
import logging
import os
import threading
import time
from functools import wraps
from timeit import default_timer as timer


# TODO better solution thatn hacky strip_result??
def stopwatch(trace_name, trace_level: int, strip_result: bool = False):
    def time_profiler(method):
        @wraps(method)
        def time_profile(*args, **kw):
            # https://stackoverflow.com/questions/7370801/how-to-measure-elapsed-time-in-python
            # perf_counter(): Return the value (in fractional seconds) of a performance counter,
            # i.e. a clock with the highest available resolution to measure a short duration.
            # It does include time elapsed during sleep and is system-wide.
            # The reference point of the returned value is undefined,
            # so that only the difference between the results of two calls is valid.
            timer_start = timer()
            # time(): Return the time in seconds since the epoch as a floating point number.
            # The specific date of the epoch and the handling of leap seconds is platform dependent.
            # On Windows and most Unix systems, the epoch is January 1, 1970, 00:00:00 (UTC) and
            # leap seconds are not counted towards the time in seconds since the epoch.
            # This is commonly referred to as Unix time. To find out what the epoch is on a given platform,
            # look at gmtime(0).
            time_start = time.time()
            # process_time(): Return the value (in fractional seconds) of the sum of the system and user CPU time
            # of the current process. It does not include time elapsed during sleep.
            # It is process-wide by definition. The reference point of the returned value is undefined,
            # so that only the difference between the results of two calls is valid.
            process_time_start = time.process_time()
            result = method(*args, **kw)
            timer_end = timer()
            time_end = time.time()
            process_time_end = time.process_time()
            global_step = -1
            if trace_name == "(6)-training_step":
                global_step = args[0].global_step

            data = {
                "trace_name": trace_name,
                "trace_level": trace_level,
                "function_name": method.__name__,
                "elapsed": (timer_end - timer_start),
                "time_start": time_start,
                "time_end": time_end,
                "process_time": (process_time_end - process_time_start),
                "process_time_start": process_time_start,
                "process_time_end": process_time_end,
                "pid": os.getpid(),
                "threading_ident": threading.get_ident(),
                "global_step": global_step,
            }
            if strip_result:
                data["len"] = result[1]
            logging.getLogger("stopwatch").debug(json.dumps(data))
            if strip_result:
                return result[0]
            return result

        return time_profile

    return time_profiler
