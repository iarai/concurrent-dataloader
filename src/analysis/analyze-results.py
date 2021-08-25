import glob
import io
import statistics as stat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.core.frame

IGNORE_LIST = ["Data size", "Warmup", "Benchmark", "Action", "iarai-playground"]

WHAT = "scratch"  # or scratch
SYNC_FILES = glob.glob(f"./benchmark_output/{WHAT}/*_sync.txt")
ASYNC_FILES = glob.glob(f"./benchmark_output/{WHAT}/*_async.txt")
FILES = sorted(ASYNC_FILES) + sorted(SYNC_FILES)


# remove wrong line breaks
def correct_misshapen_lines(file_path: str) -> None:
    __file_text = ""
    with open(file_path, "r") as f:
        __file_text = f.read()

    __file_text = __file_text.replace("ms((", "ms\n((")
    with open(file_path, "w") as f:
        f.write(__file_text)


def plot_all(time_dict, plot_max=True, log_scale=True, title=None):
    pos_x = 0
    pos_y = 0
    titles = ["getitem", "fetch", "worker", "load_all", "benchmark_scratch_dataloader", "2-1"]
    _, ax = plt.subplots(3, 2)
    for action in range(5):
        means = []
        medians = []
        mins = []
        maxs = []
        labels = []
        for e in time_dict:
            current = time_dict[e]
            # actions: get_item, fetch, _worker_loop,
            _min, _max, _mean, _median = (
                min(current[action]),
                max(current[action]),
                stat.mean(current[action]),
                stat.median(current[action]),
            )
            mins.append(_min)
            if plot_max:
                maxs.append(_max)
            means.append(_mean)
            medians.append(_median)
            labels.append(e.split("/")[-1])

        x = np.arange(len(labels))
        width = 0.2

        _ = ax[pos_x, pos_y].bar(x + width * 1 + 0.1, means, width, label=f"mean: {_mean:.2f}ms, {_mean / 60000:.2f} m")
        _ = ax[pos_x, pos_y].bar(
            x + width * 2 + 0.1, medians, width, label=f"median: {_median:.2f}ms, {_median / 60000:.2f} m"
        )
        _ = ax[pos_x, pos_y].bar(x + width * 3 + 0.1, mins, width, label=f"min: {_min:.2f}ms, {_min / 60000:.2f} m")
        if plot_max:
            _ = ax[pos_x, pos_y].bar(x + width * 4 + 0.1, maxs, width, label=f"max: {_max:.2f}ms, {_max / 60000:.2f} m")

        ax[pos_x, pos_y].set_ylabel(titles[action])
        if title is not None:
            ax[pos_x, pos_y].set_title(title)

        ax[pos_x, pos_y].set_xticks(x)
        ax[pos_x, pos_y].set_xticklabels(labels, rotation=45, fontsize=8, ha="center")
        ax[pos_x, pos_y].legend(prop={"size": 8})
        ax[pos_x, pos_y].grid(which="both")
        ax[pos_x, pos_y].grid(which="minor", alpha=0.5, linestyle="--")
        if log_scale:
            ax[pos_x, pos_y].set_yscale("log")

        pos_x += 1
        if pos_x >= 3:
            pos_x = 0
            pos_y += 1
    # add fetch - get_item

    plt.show()


def get_csv(file_path: str) -> pandas.core.frame.DataFrame:
    __lines_list = []
    with open(file_path, "r") as f:
        for _, line in enumerate(f):
            if not any(ignore_word in line for ignore_word in IGNORE_LIST):
                __lines_list.append(line)
    __joined_line = io.StringIO("\n".join(__lines_list))
    return pd.read_csv(__joined_line, sep=" ", low_memory=False, error_bad_lines=False, index_col=False)


if __name__ == "__main__":
    working_file_path = FILES[0]

    time_dict = {}
    for working_file_path in FILES:
        print(f"Working with {working_file_path}")
        # change lines containing ms(( -> ms \n ((
        correct_misshapen_lines(working_file_path)

        # get the CSV object
        csv_df = get_csv(working_file_path)
        csv_df.columns = ["function", "function_name", "id", "function_id", "time", "ms"]  # noqa

        all_times = []
        r = csv_df.query('function_name == "__getitem__"')
        print(f"Get item {len(r)}")
        all_times.append(r.time)
        r = csv_df.query('function_name == "fetch"')
        print(f"Fetch {len(r)}")
        all_times.append(r.time)
        r = csv_df.query('function_name == "_worker_loop"')
        print(f"Worker loop {len(r)}")
        all_times.append(r.time)
        r = csv_df.query('function_name == "load_all"')
        print(f"load all {len(r)}")
        all_times.append(r.time)
        r = csv_df.query('function_name == "benchmark_scratch_dataloader"')
        print(f"benchmark_scratch_dataloader {len(r)}")
        all_times.append(r.time)
        time_dict[working_file_path] = all_times

    plot_all(time_dict, plot_max=True, log_scale=True)
