import argparse
import json
import logging
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_all(time_dict, plot_max=True, log_scale=True):
    titles = ["getitem", "fetch", "worker", "load_all", "benchmark_dataloader", "2-1"]
    rows = 5
    fig, ax = plt.subplots(rows, len(time_dict) * 2)
    for action in range(5):
        pos_x = action
        for i, e in enumerate(time_dict):
            current = time_dict[e]
            # actions: get_item, fetch, _worker_loop,
            _min, _max, _mean, _median = (
                current[action].min(),
                current[action].max(),
                current[action].mean(),
                current[action].median(),
            )
            labels = ["min", "max", "mean", "median"]
            mins = _min
            means = _mean
            medians = _median
            maxs = _max

            x = np.arange(len(labels))
            width = 0.2

            _ = ax[pos_x, i * 2].bar(0, means, width, label=f"mean: {_mean:.2f}ms, {_mean / 60000:.2f} m")
            _ = ax[pos_x, i * 2].bar(1, medians, width, label=f"median: {_median:.2f}ms, {_median / 60000:.2f} m")
            _ = ax[pos_x, i * 2].bar(2, mins, width, label=f"min: {_min:.2f}ms, {_min / 60000:.2f} m")
            if plot_max:
                _ = ax[pos_x, i * 2].bar(3, maxs, width, label=f"max: {_max:.2f}ms, {_max / 60000:.2f} m")

            ax[pos_x, i * 2].set_ylabel(titles[action])
            ax[pos_x, i * 2].set_title(e)

            ax[pos_x, i * 2].set_xticks(x)
            ax[pos_x, i * 2].set_xticklabels(labels, rotation=45, fontsize=8, ha="center")
            ax[pos_x, i * 2].legend(prop={"size": 8})
            ax[pos_x, i * 2].grid(which="both")
            ax[pos_x, i * 2].grid(which="minor", alpha=0.5, linestyle="--")
            if log_scale:
                ax[pos_x, i * 2].set_yscale("log")

            if len(current[action]) == 0:
                continue
            violin = ax[pos_x, i * 2 + 1].violinplot(
                current[action].tolist(),
                vert=False,
                widths=1.1,
                showmeans=True,
                showextrema=True,
                showmedians=True,
                quantiles=[0.25, 0.75],
                bw_method=0.5,
            )
            violin["cmeans"].set_color("r")
            violin["cmedians"].set_linewidth(2)
            violin["cmaxes"].set_color("grey")
            violin["cmaxes"].set_linestyle(":")
            violin["cmins"].set_color("grey")
            violin["cmins"].set_linestyle(":")

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


def handle_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_folder", type=Path, default=Path("/iarai/work/logs/storage_benchmarking"))
    return parser


def main(*args):
    parser = handle_arguments()
    args = parser.parse_args(args)
    # TODO make configurable in cli,
    # TODO read metadata as well
    # TODO filter on metadata as well
    files = args.output_base_folder.rglob("**/results-*.log")
    data_grouped_by_dir = {}
    for working_file_path in files:
        results = parse_results_log(working_file_path)
        with (working_file_path.parent / "metadata.json").open("r") as f:
            metadata = json.load(f)  # noqa
        # TODO refine grouping
        key = "_".join(working_file_path.parent.name.split("_")[1:])
        if len(results) == 0:
            continue
        data_grouped_by_dir.setdefault(key, []).extend(results)
    time_dict = {}

    for dir, data in data_grouped_by_dir.items():
        time_dict[dir] = group_by_function_name(pd.DataFrame.from_records(data=data))

    plot_all(time_dict, plot_max=True, log_scale=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
