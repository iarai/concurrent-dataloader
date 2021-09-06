import json
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_all(time_dict, plot_max=True, log_scale=True, title=None):
    pos_x = 0
    pos_y = 0
    titles = ["getitem", "fetch", "worker", "load_all", "benchmark_dataloader", "2-1"]
    fig, ax = plt.subplots(3, 2)
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
                current[action].min(),
                current[action].max(),
                current[action].mean(),
                current[action].median(),
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

    # pretty output
    fig.subplots_adjust(top=0.980, bottom=0.100, left=0.045, right=0.980, hspace=0.415, wspace=0.130)
    # add fetch - get_item

    plt.show()


def group_by_function_name(csv_df: pd.DataFrame) -> List[pd.DataFrame]:
    all_times = []
    r = csv_df.query('function_name == "__getitem__"')
    logging.info(f"Get item {len(r)}")
    all_times.append(r.process_time)
    r = csv_df.query('function_name == "fetch"')
    logging.info(f"Fetch {len(r)}")
    all_times.append(r.process_time)
    r = csv_df.query('function_name == "_worker_loop"')
    logging.info(f"Worker loop {len(r)}")
    all_times.append(r.process_time)
    r = csv_df.query('function_name == "load_all"')
    logging.info(f"load all {len(r)}")
    all_times.append(r.process_time)
    r = csv_df.query('function_name == "benchmark_dataloader"')
    logging.info(f"benchmark_dataloader {len(r)}")
    all_times.append(r.process_time)
    return all_times


def parse_results_log(working_file_path: str) -> List[Dict]:
    with open(working_file_path, "r") as f:
        skipped_lines_count = 0
        data = []
        for line in f:
            try:
                data.append(json.loads(line))
            except JSONDecodeError:
                skipped_lines_count += 1
    if skipped_lines_count > 0:
        logging.warning("Skipped %s lines while reading %s", skipped_lines_count, working_file_path)
    return data


def main():
    # TODO make configurable in cli,
    # TODO read metadata as well
    # TODO filter on metadata as well
    files = Path(".").rglob("**/results-*.log")
    data_grouped_by_dir = {}
    for working_file_path in files:
        with (working_file_path.parent / "metadata.json").open("r") as f:
            metadata = json.load(f)
        # TODO refine grouping
        key = metadata["action"] + "_" + "_".join(metadata["args"])
        data_grouped_by_dir.setdefault(key, []).extend(parse_results_log(working_file_path))

    time_dict = {}
    for dir, data in data_grouped_by_dir.items():
        time_dict[dir] = group_by_function_name(pd.DataFrame.from_records(data=data))

    plot_all(time_dict, plot_max=True, log_scale=True)


if __name__ == "__main__":
    main()
