import io
import json
import logging
import re
from datetime import timedelta
from json import JSONDecodeError
from pathlib import Path
from typing import Dict
from typing import List

import humanize
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from pandas import DataFrame
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict


def plot_all(df: DataFrame, function_name: str, group_by: List[str], plot_max=True, log_scale=True, figsize=(50, 50)):
    function_names = ["__getitem__", "fetch", "_worker_loop", "load_all", "benchmark_dataloader"]
    assert function_name in function_names

    df = df.groupby(group_by)
    num_rows = len(df.groups)
    num_cols = 4

    if num_rows == 0:
        return

    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, (key, df) in enumerate(df):
        df_for_function_name = df[df["function_name"] == function_name]
        elapased = df_for_function_name["elapsed"]
        if len(elapased) == 0:
            continue

        _min, _max, _mean, _median = (
            elapased.min(),
            elapased.max(),
            elapased.mean(),
            elapased.median(),
        )
        labels = ["mean", "median", "min", "max"]
        mins = _min
        means = _mean
        medians = _median
        maxs = _max

        x = np.arange(len(labels))
        width = 0.2
        pos_x = i
        pos_y_start = 0
        _ = ax[pos_x, pos_y_start].bar(
            0, means, width, label=f"mean: {_mean:.2f}ms, {_mean / 60000:.2f} m", color="red"
        )
        _ = ax[pos_x, pos_y_start].bar(
            1, medians, width, label=f"median: {_median:.2f}ms, {_median / 60000:.2f} m", color="green"
        )
        _ = ax[pos_x, pos_y_start].bar(
            2, mins, width, label=f"min: {_min:.2f}ms, {_min / 60000:.2f} m", color="lightblue"
        )
        if plot_max:
            _ = ax[pos_x, pos_y_start].bar(
                3, maxs, width, label=f"max: {_max:.2f}ms, {_max / 60000:.2f} m", color="blue"
            )

        ax[pos_x, pos_y_start].set_ylabel(function_name)
        ax[pos_x, pos_y_start].set_title(key)

        ax[pos_x, pos_y_start].set_xticks(x)
        ax[pos_x, pos_y_start].set_xticklabels(labels, rotation=45, fontsize=8, ha="center")
        ax[pos_x, pos_y_start].legend(prop={"size": 8})
        ax[pos_x, pos_y_start].grid(which="both")
        ax[pos_x, pos_y_start].grid(which="minor", alpha=0.5, linestyle="--")
        if log_scale:
            ax[pos_x, pos_y_start].set_yscale("log")

        violin = ax[pos_x, pos_y_start + 1].violinplot(
            elapased.tolist(),
            vert=False,
            widths=1.1,
            showmeans=True,
            showextrema=True,
            showmedians=True,
            quantiles=[0.25, 0.75],
            bw_method=0.5,
        )
        violin["cmeans"].set_color("red")
        violin["cmedians"].set_linewidth(2)
        violin["cmedians"].set_color("green")
        violin["cmaxes"].set_color("blue")
        violin["cmaxes"].set_linestyle(":")
        violin["cmins"].set_color("lightblue")
        violin["cmins"].set_linestyle(":")
        ax[pos_x, pos_y_start + 1].set_xlabel("elapsed [s]")
        ax[pos_x, pos_y_start + 1].set_ylabel("kernel density [-]")

        df_grouped = df_for_function_name.groupby("run").agg(
            {"time_start": "min", "time_end": "max", "len": "sum", **{k: "first" for k in group_by}}
        )
        df_grouped["elapsed [s]"] = df_grouped["time_end"] - df_grouped["time_start"]
        df_grouped["throughput [Bytes/s]"] = df_grouped["len"] / df_grouped["elapsed [s]"] / 10 ** 6
        df_grouped["throughput [Mbit/s]"] = df_grouped["len"] / df_grouped["elapsed [s]"] / 10 ** 6 * 8
        df_grouped[["throughput [Mbit/s]"]].boxplot(ax=ax[pos_x, pos_y_start + 2])
        df_grouped[["elapsed [s]"]].boxplot(ax=ax[pos_x, pos_y_start + 3])
    # pretty output
    fig.subplots_adjust(top=0.980, bottom=0.100, left=0.045, right=0.980, hspace=0.415, wspace=0.130)


def get_run_stats(df: DataFrame, group_by: List[str], row_filter: Dict[str, List[str]] = None):
    if row_filter is not None:
        for col, items in row_filter.items():
            df = df.filter(index=col, items=items, axis=0)
    df = df.groupby(group_by + ["run"]).agg(
        **{"downloaded data [B]": ("len", "sum"), "time_start": ("time_start", "min"), "time_end": ("time_end", "max"),}
    )
    df["total_elpased_time [s]"] = df["time_end"] - df["time_start"]
    df["downloaded data [MB]"] = df["downloaded data [B]"] / 10 ** 6
    df["throughput [MBit/s]"] = df["downloaded data [MB]"] * 8 / df["total_elpased_time [s]"]

    return df


def get_throughputs(df: DataFrame, group_by: List[str], row_filter: Dict[str, List[str]] = None):
    df = get_run_stats(df, group_by=group_by, row_filter=row_filter)
    df = df.groupby(group_by).agg(
        **{
            "total_elpased_time [s]": ("total_elpased_time [s]", "sum"),
            "downloaded data [MB]": ("downloaded data [MB]", "sum"),
            "med. throughput [MBit/s]": ("throughput [MBit/s]", "median"),
            "avg. throughput [MBit/s]": ("throughput [MBit/s]", "mean"),
            "min. throughput [MBit/s]": ("throughput [MBit/s]", "min"),
            "max. throughput [MBit/s]": ("throughput [MBit/s]", "max"),
        }
    )
    df["throughput [MBit/s]"] = df["downloaded data [MB]"] * 8 / df["total_elpased_time [s]"]
    return df


def get_thread_stats(df: DataFrame, group_by: List[str], trace_level=5):
    s = (
        df[df["trace_level"] == trace_level]
        .groupby("threading_ident")
        .agg(
            **{
                "time_start_thread": ("time_start", "min"),
                "time_end_thread": ("time_end", "max"),
                "total_elapsed_thread": ("elapsed", sum),
                **{k: (k, "first") for k in group_by},
            }
        )
    )
    s["elapsed_thread"] = s["time_end_thread"] - s["time_start_thread"]
    s["elapsed_processing"] = s["total_elapsed_thread"] / s["elapsed_thread"]
    return s


# https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """Lightens the given color by multiplying (1-luminosity) by the given
    amount. Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:  # noqa
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_throughput_per_storage(df, group_by: List[str]):
    collected = {}

    x_label = group_by[1]

    df_for_function_name = df[df["function_name"] == "__getitem__"]
    df_for_function_name["request_time"] = df_for_function_name["time_end"] - df_for_function_name["time_start"]
    nodes = set(df_for_function_name["node"].drop_duplicates().tolist())
    df_grouped_by_run = df_for_function_name.groupby(["run"]).agg(
        {"time_start": "min", "time_end": "max", "len": "sum", **{k: "first" for k in group_by}}
    )
    df_grouped_by_run["runtime"] = df_grouped_by_run["time_end"] - df_grouped_by_run["time_start"]
    df_grouped_by_run["throughput [Mbit/s]"] = df_grouped_by_run["len"] / df_grouped_by_run["runtime"] / 10 ** 6 * 8

    # sum of all runtimes and all downloaded data by summer over runs
    df_aggregated_over_runs = df_grouped_by_run.groupby(group_by).agg(
        **{
            "len": ("len", "sum"),
            "runtime": ("runtime", "sum"),
            "min_throughput": ("throughput [Mbit/s]", "min"),
            "max_throughput": ("throughput [Mbit/s]", "max"),
            **{k: (k, "first") for k in group_by},
        }
    )

    # https://en.wikipedia.org/wiki/Data-rate_units#Megabit_per_second
    df_aggregated_over_runs["throughput [Mbit/s]"] = (
        df_aggregated_over_runs["len"] / df_aggregated_over_runs["runtime"] / 10 ** 6 * 8
    )
    df_aggregated_over_runs["min_throughput"] = df_aggregated_over_runs["min_throughput"]

    # all requests from all data
    df_aggregated_over_requests = df_for_function_name.groupby(group_by).agg(
        **{
            "min_request_time": ("request_time", "min"),
            "max_request_time": ("request_time", "max"),
            "median_request_time": ("request_time", "median"),
            "mean_request_time": ("request_time", "mean"),
            **{k: (k, "first") for k in group_by},
        }
    )

    fig, ax1 = plt.subplots(figsize=(50, 10))
    ax1.set_ylim([0, max(200, df_grouped_by_run["throughput [Mbit/s]"].max())])
    ax2 = ax1.twinx()
    cmap = plt.cm.get_cmap("Set1")

    result = []
    # TODO loop over groups instead
    for i, dataset in enumerate(["s3", "scratch"]):
        df = df_aggregated_over_runs[df_aggregated_over_runs["dataset"] == dataset]

        storage = dataset

        ax1.plot(
            df[x_label],
            df["throughput [Mbit/s]"],
            linewidth=4,
            color=cmap(i),
            label=f"{storage} throughput over all runs, min/max hull per run",
        )
        ax1.fill_between(
            df[x_label], df["min_throughput"], df["max_throughput"], color=lighten_color(cmap(i), 0.2), alpha=0.5
        )
        ax1.set_xticks(df[x_label])

        df2 = df_aggregated_over_requests[df_aggregated_over_requests["dataset"] == dataset]
        ax2.plot(
            df2[x_label],
            df2["median_request_time"],
            linewidth=4,
            color=cmap(i),
            label=f"{storage} request_time",
            linestyle="dashed",
        )
        result.append(
            {
                "dataset": dataset,
                "throughput": df["throughput [Mbit/s]"],
                "median_request_time": df2["median_request_time"],
            }
        )

    fig.legend(handlelength=5)

    ax1.set_ylabel("throughput [Mbit/s]")
    ax1.set_xlabel(f"{x_label} [#processes]")
    ax1.set_title(f"Storage benchmarking {nodes} {list(collected.keys())}")
    ax2.set_ylabel("Request time [s]")
    return result


def plot_events_timeline(
    df_dataloader, color_column: str = "threading_ident", cycle=11, summary_only=False, verbose=True,
):
    df_dataloader = df_dataloader.sort_values(
        ["pid", "trace_level", "threading_ident", "time_start"], ascending=[False, False, False, False]
    ).reset_index(drop=True)

    total_elapsed = df_dataloader["time_end"].max() - df_dataloader["time_start"].min()
    total_bytes = df_dataloader["len"].sum()
    overall_rate_mbps = {humanize.naturalsize(total_bytes / total_elapsed)}
    overall_rate_mbitps = {humanize.naturalsize(total_bytes / total_elapsed * 8)}
    if verbose:
        print(f"total_elapsed={timedelta(seconds=total_elapsed)}")
        print(f"total_bytes={humanize.naturalsize(total_bytes)}")
        print(f"overall rate {overall_rate_mbps}/s")
        print(f"overall rate {overall_rate_mbitps}it/s")

    if summary_only:
        return overall_rate_mbps, overall_rate_mbitps

    dict_dataloader = df_dataloader.to_dict("index")
    threading_idents = {d[color_column] for d in dict_dataloader.values()}

    cmap = plt.cm.get_cmap("hsv", cycle)
    all_colors = [cmap(i) for i in range(cycle)]

    threading_ident_cmap = {
        threading_ident: all_colors[i % cycle] for i, threading_ident in enumerate(threading_idents)
    }
    lines = [[(d["time_start"], i), (d["time_end"], i)] for i, (k, d) in enumerate(dict_dataloader.items())]
    c = [threading_ident_cmap[d[color_column]] for d in dict_dataloader.values()]

    lc = matplotlib.collections.LineCollection(lines, colors=c, linewidths=2)
    fig, ax = plt.subplots(figsize=(50, 50))
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)


def plot_events_timeline_detailed(
    df_dataloader, color_column: str = "threading_ident", highlight_thread: int = None, filter_function: str = None,
):
    if filter_function is not None:
        df_dataloader = df_dataloader[df_dataloader["function_name"] == filter_function]

    color_list = {}
    thread_ids = np.array(list(df_dataloader["threading_ident"]))
    pallete = sns.color_palette(None, len(np.unique(thread_ids)))
    thread_runtimes = {}
    for index, t in enumerate(np.unique(thread_ids)):
        color_list[t] = pallete[index]
        thread_runtimes[t] = max(df_dataloader[df_dataloader["threading_ident"] == t]["time_end"]) - min(
            df_dataloader[df_dataloader["threading_ident"] == t]["time_start"]
        )
    df_dataloader = df_dataloader.sort_values(
        ["pid", "threading_ident", "trace_level", "time_start"], ascending=[False, False, False, False]
    ).reset_index(drop=True)

    min_time = min(df_dataloader["time_start"])
    max_time = max(df_dataloader["time_end"]) - min_time

    dict_dataloader = df_dataloader.to_dict("index")
    print(len(dict_dataloader.items()))

    colors = []
    lines = []
    texts = []
    for _, t in enumerate(np.unique(thread_ids)):
        last_i = 0
        for i, (_, param_series) in enumerate(dict_dataloader.items()):
            if param_series["threading_ident"] == t:
                last_i = i
                lines.append([(param_series["time_start"] - min_time, i), (param_series["time_end"] - min_time, i)])
                if highlight_thread is not None:
                    if param_series["threading_ident"] == highlight_thread:
                        colors.append("black")
                    else:
                        colors.append("red")
                else:
                    colors.append(color_list[param_series["threading_ident"]])
        lines.append([(0, last_i), (max_time, last_i)])
        texts.append((thread_runtimes[t], 0, last_i))
        colors.append("silver")

    print(f"Lines num: {len(lines)}")

    lc = matplotlib.collections.LineCollection(lines, colors=colors, linewidths=2)
    fig, ax = plt.subplots(figsize=(50, 50))
    ax.add_collection(lc)
    for i in texts:
        ax.text(i[1], i[2], i[0])
    ax.autoscale()
    ax.margins(0.1)


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
        logging.warning("Skipped %s lines while readinget_thread_statsg %s", skipped_lines_count, working_file_path)
    return data


def extract_pandas(
    output_base_folder: Path, folder_filter: str = "**", filter_by_metadata: Dict[str, List[str]] = None,
):
    files = list(output_base_folder.rglob(f"{folder_filter}/results-*.log"))
    data = []
    for working_file_path in tqdm.tqdm(files, total=len(files)):
        results = parse_results_log(working_file_path)
        if len(results) == 0:
            continue
        with (working_file_path.parent / "metadata.json").open("r") as f:
            metadata = json.load(f)
        if filter_by_metadata is not None:
            for k, v in filter_by_metadata.items():
                if metadata[k] not in v:
                    continue
        results = pd.DataFrame.from_records(data=results)

        for k, v in metadata.items():
            if not isinstance(v, (int, float, complex)):
                results[k] = str(v)
            else:
                results[k] = v

        results["source_file"] = working_file_path
        results["run"] = working_file_path.parent.name

        # filter out old data format missing dataset etc.
        if "dataset" not in results.columns:
            continue

        data.append(results)
    df = pd.concat(data)
    df.groupby
    return df


def extract_gpu_utilization(output_base_folder: Path, folder_filter: str = "**", device_id=0):
    folders = list(output_base_folder.rglob(f"{folder_filter}"))
    files = output_base_folder.rglob(f"{folder_filter}/lightning/default/version_0")
    data = pd.DataFrame()
    for working_file_path in tqdm.tqdm(files, total=len(folders)):
        event_acc = EventAccumulator(str(working_file_path))
        event_acc.Reload()
        w_times, step_nums, vals = zip(*event_acc.Scalars(f"device_id: {device_id}/utilization.gpu (%)"))
        data = data.append(
            {
                "run": working_file_path.parent.parent.parent.name,
                "gpu": np.array(vals),
                "gpu_mean": np.array(vals).mean(),
                "gpu_median": np.median(np.array(vals)),
                "std": np.std(np.array(vals)),
            },
            ignore_index=True,
        )
    return data


def extract_gpuutil(
    output_base_folder: Path, folder_filter: str = "**", filter_by_metadata: Dict[str, List[str]] = None,
):
    files = list(output_base_folder.rglob(f"{folder_filter}/gpuutil-*.log"))
    data = []
    for working_file_path in tqdm.tqdm(files, total=len(files)):
        results = parse_results_log(working_file_path)
        if len(results) == 0:
            continue
        header = []
        header.append("timestamp")
        for i in results[0]["gpu_data"]:
            header.append(f"gpu_util_{i}")
            header.append(f"mem_util_{i}")
        header.append("run")
        lines = []
        for result in results:
            line = []
            line.append(result["timestamp"])
            for item in result["gpu_data"]:
                line.append(result["gpu_data"][item]["gpu_util"])
                line.append(result["gpu_data"][item]["mem_util"])
            line.append(working_file_path.parent.name)
            lines.append(line)
        results = pd.DataFrame.from_records(lines)
        data.append(results)
    df = pd.concat(data)
    df.columns = header
    df.groupby
    df.sort_values(["timestamp"], ascending=True)
    return df


def extract_profiling(output_base_folder: Path, folder_filter: str = "**", device_id=0):
    folders = list(output_base_folder.rglob(f"{folder_filter}"))
    files = output_base_folder.rglob(f"{folder_filter}/lightning/*.txt")
    print(files)
    data = pd.DataFrame()
    column_names = ["run", "function", "mean_duration", "num_calls", "total_time", "percentage", "NaN"]
    for working_file_path in tqdm.tqdm(files, total=len(folders)):
        with open(working_file_path) as file:
            lines = file.readlines()
            lines = [re.sub(r"[\n\t\s]*", "", f"{working_file_path.parent.parent.name}|" + line) for line in lines[6:]]
            text = "\n".join(lines)
            data = data.append(pd.read_csv(io.StringIO(text), sep="|", header=None, names=column_names))
    data.drop("NaN", axis=1, inplace=True)
    return data


def extract_timelines(
    output_base_folder: Path, folder_filter: str = "**", filter_by_metadata: Dict[str, List[str]] = None,
):
    files = list(output_base_folder.rglob(f"{folder_filter}/timeline-*.log"))
    data = []
    for working_file_path in tqdm.tqdm(files, total=len(files)):
        results = parse_results_log(working_file_path)
        if len(results) == 0:
            continue
        results = pd.DataFrame.from_records(data=results)
        results = pd.merge(
            results[results["end_time"].isnull()], results[results["start_time"].isnull()], left_on="id", right_on="id"
        ).drop(["end_time_x", "start_time_y"], axis=1)
        results["source_file"] = working_file_path
        results["run"] = working_file_path.parent.name
        # filter out old data format missing dataset etc.
        data.append(results)
    df = pd.concat(data)
    df.groupby
    return df


def show_timelines(df, run, lanes, colors, flat=False, zoom=False, zoom_epochs=1):
    fig, ax = plt.subplots(figsize=(30, 20))
    start = min(df["start_time_x"])
    end = max(df["end_time_y"])

    total_runtime = end - start
    number_of_epochs = 20

    if zoom:
        df = df[df["start_time_x"] < start + ((total_runtime / number_of_epochs) * zoom_epochs)]

    i = 0
    for _, row in df.sort_values(["start_time_x"], ascending=True).iterrows():
        duration = row["end_time_y"] - row["start_time_x"]
        x1 = row["start_time_x"] - start
        if duration < 0.08:
            duration = 0.1
        x2 = x1 + duration
        if not flat:
            lane = i
            i += 1
        else:
            lane = lanes[row["item_x"]]
        ax.plot([x1, x2], [lane, lane], color=colors[row["item_x"]], label=row["item_x"], linewidth=1)
    ax.set_xlabel("Experiment duration", loc="center")
    ax.set_ylabel("Item", loc="top")
    # ['20211109f152412', 'benchmark', 'e2e', 'lightning', 's3', '256', '4', '16', '1', 'vanilla', 'sync']  # noqa
    filename = run.split("_")
    ax.set_title(
        f"Runtime for each function, impl: {filename[9]},"
        f" cache: {filename[8]}, "
        f" batch size: {filename[5]}, "
        f" lib: {filename[3]}",
        loc="center",
    )
    ax.legend()
    ax.grid(linestyle="--", which="both")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Put a legend below current axis
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.show()


def show_timelines_with_gpu(df, gpu_util, lanes, colors, run, flat=False, show_gpu=False, zoom=False, zoom_epochs=1):
    fig, ax = plt.subplots(figsize=(30, 25))
    plt.rcParams.update({"font.size": 18})
    start = min(df["start_time_x"])
    end = max(df["end_time_y"])
    gpu_start = min(gpu_util["timestamp"])

    total_runtime = end - start
    number_of_epochs = 20

    if zoom:
        df = df[df["start_time_x"] < start + ((total_runtime / number_of_epochs) * zoom_epochs)]
        gpu_util = gpu_util[gpu_util["timestamp"] < gpu_start + ((total_runtime / number_of_epochs) * zoom_epochs)]

    lane = 0
    for _, row in df.sort_values(["start_time_x"], ascending=True).iterrows():
        duration = row["end_time_y"] - row["start_time_x"]
        x1 = row["start_time_x"] - start
        if duration < 0.15:
            duration = 0.2
        x2 = x1 + duration
        if not flat:
            lane += 10
        else:
            lane = lanes[row["item_x"]]
        ax.plot([x1, x2], [lane, lane], color=colors[row["item_x"]], label=row["item_x"], linewidth=3)

    ax.set_xlabel("Experiment duration (S)", loc="right")
    ax.set_ylabel("Operation activity lane", loc="center")
    filename = run.split("_")

    ax.legend()
    ax.grid(linestyle="--", which="both")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # Put a legend below current axis
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
    )

    gpu_util_mean_no_zeros = 0

    if show_gpu:
        ax2 = ax.twinx()
        ax2.set_ylabel("GPU/Memory utilization (green, maroon) [%]")

        r"{\fontsize{50pt}{3em}\selectfont{}a}{\fontsize{20pt}{3em}\selectfont{}N"
        ax2.set_ylim([-3, 103])
        gpu_events = []
        for i in gpu_util["timestamp"]:
            gpu_events.append(i - start)
        ax2.plot(gpu_events, gpu_util["gpu_util_2"], color="cyan", linestyle="--", linewidth=2)
        ax2.plot(gpu_events, gpu_util["mem_util_2"], color="maroon", linestyle="--", linewidth=2)
        gpu_util_zeros = (len(gpu_util[gpu_util["gpu_util_2"] == 0]["gpu_util_2"]) / len(gpu_util["gpu_util_2"])) * 100
        gpu_util_mean_no_zeros = np.mean(gpu_util[gpu_util["gpu_util_2"] > 0]["gpu_util_2"])
        mem_util_mean = np.mean(gpu_util["mem_util_2"])
        mem_util_mean_no_zeros = np.mean(gpu_util[gpu_util["mem_util_2"] > 0]["mem_util_2"])
        ax2.plot(
            gpu_events, [gpu_util_mean_no_zeros] * len(gpu_events), label="GPU Util Mean", linewidth=2, color="cyan"
        )
        ax2.plot(
            gpu_events, [mem_util_mean_no_zeros] * len(gpu_events), label="Mem Util Mean", linewidth=2, color="maroon"
        )
        print(gpu_util_mean_no_zeros, mem_util_mean_no_zeros)
        ax2.legend()

    ax.set_title(
        f"Total runtime per operation \n Implementation: {filename[9]},"
        # f" use cache: {filename[8]}, "
        f" batch size: {filename[5]}, "
        f" library: {filename[3]}, "
        f"\n GPU unused: {round(gpu_util_zeros, 2)} %, "
        f" mean GPU usage: {round(gpu_util_mean_no_zeros, 2)} %",
        loc="center",
    )

    plt.show()
    return {
        "runtime": end - start,
        "gpu_util_zero": gpu_util_zeros,
        "gpu_util_mean_no_zeros": gpu_util_mean_no_zeros,
        "mem_util_mean": mem_util_mean,
        "mem_util_mean_no_zeros": mem_util_mean_no_zeros,
        "implementation": filename[9],
        "cache": filename[8],
        "library": filename[3],
    }


def get_gpu_stats(df, gpu_util, run, flat=False, show_gpu=False, zoom=False, zoom_epochs=1):
    start = min(df["start_time_x"])
    end = max(df["end_time_y"])
    filename = run.split("_")

    gpu_events = []
    for i in gpu_util["timestamp"]:
        gpu_events.append(i - start)
    gpu_util_zeros = (len(gpu_util[gpu_util["gpu_util_2"] == 0]["gpu_util_2"]) / len(gpu_util["gpu_util_2"])) * 100
    gpu_util_mean_no_zeros = np.mean(gpu_util[gpu_util["gpu_util_2"] > 0]["gpu_util_2"])
    mem_util_mean = np.mean(gpu_util["mem_util_2"])
    mem_util_mean_no_zeros = np.mean(gpu_util[gpu_util["mem_util_2"] > 0]["mem_util_2"])

    return {
        "runtime": end - start,
        "gpu_util_zero": gpu_util_zeros,
        "gpu_util_mean_no_zeros": gpu_util_mean_no_zeros,
        "mem_util_mean": mem_util_mean,
        "mem_util_mean_no_zeros": mem_util_mean_no_zeros,
        "implementation": filename[9],
        "cache": filename[8],
        "library": filename[3],
    }


def plot_histogram(throughput, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({"font.size": 12})
    ax.hist(throughput, bins=3)
    mean = np.mean(throughput).round(2)
    std = np.std(throughput).round(2)
    ax.axvline(np.mean(throughput), linestyle="dashed", linewidth=2, color="green")
    ax.set_title(title + f": mean = {mean}," f" var: {np.var(throughput).round(2)}," f" std:{std}")
    ax.set_xlabel("Throughput (imgs/S)", loc="right")
    ax.set_ylabel("Experiments", loc="center")
    ax.set_xlim(mean - 3 * std, mean + 3 * std)
    ax.plot()


def plot_all_histograms(res, impls, libs, display=True):
    df_throughput_all = pd.DataFrame()
    for impl in impls:
        for lib in libs:
            throughput = []
            key = f"{impl}_{lib}"
            for experiment in range(len(res)):
                data = res[experiment].round(2)
                data = data[(data["library"] == lib) & (data["implementation"] == impl)]["throughput"]
                throughput.append(data)
            df_throughput_all[key] = pd.DataFrame.from_records(throughput)
            if display:
                plot_histogram(throughput, f"Setup: {impl}, {lib}")
    return df_throughput_all


def plot_violins(throughput, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams.update({"font.size": 12})
    all_data = []
    labels = []
    for i in range(len(throughput)):
        data = throughput.iloc[i]
        index = data.name
        data = data.values.tolist()
        all_data.append(data[:-3])
        labels.append(index)
    ax.violinplot(all_data, vert=True, widths=0.5, showmeans=True)

    ax.xaxis.set_tick_params(direction="out")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels=labels, rotation=20)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel("Experiment setup")
    ax.set_ylabel("Throughput (imgs/s)")
    ax.set_title(title)
    ax.grid(linestyle="--", which="both")
    plt.plot()
