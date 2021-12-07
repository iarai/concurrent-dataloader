import glob
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch.cuda
from benchmarking.analysis.analyze_results import parse_results_log
from dataset.scratch_dataset import ScratchDataset


# TODO add s3 etc. if available
# TODO test mp and without
@pytest.mark.parametrize(
    "dataset", [("scratch")],
)
def test_benchmark_scratch_dataset(dataset: str):
    from benchmark.benchmark_dataset import main

    with tempfile.TemporaryDirectory() as temp_dir:
        index_file = os.path.join(temp_dir, "index_scratch.json")

        # TODO relative path from src file may not be safe if we add setup.py - use importlib.resources instead.
        ScratchDataset.index_all(Path(os.path.dirname(os.path.abspath(__file__))) / "resources", file_name=index_file)

        output_base_folder = Path(temp_dir) / "benchmark_results"
        output_base_folder.mkdir(exist_ok=False, parents=True)

        # TODO #17 control limit amount of data here instead of relying on hard-coded assumptions in main
        main("-a", dataset, "--output_base_folder", str(output_base_folder), "-args", index_file)
        results_files = glob.glob(f"{output_base_folder}/**/results-*.log", recursive=True)
        assert len(results_files) == 9, results_files
        for f in results_files:
            parse_results_log(f)
        metadata_files = glob.glob(f"{output_base_folder}/**/*.json", recursive=True)
        assert len(metadata_files) == 2, metadata_files
        for f in metadata_files:
            with open(f, "r") as f:
                json.load(f)


@pytest.mark.parametrize(
    "batch_size,num_workers,data_loader_type,repeat", [(2, 5, "sync", 1)],
)
def test_benchmark_dataloader(batch_size: int, num_workers: int, data_loader_type: str, repeat: int):
    from benchmark.benchmark_dataloader import main

    with tempfile.TemporaryDirectory() as temp_dir:

        output_base_folder = Path(temp_dir) / "benchmark_results"
        output_base_folder.mkdir(exist_ok=False, parents=True)

        main(
            "--batch_size",
            str(batch_size),
            "--num_workers",
            str(num_workers),
            "--data_loader_type",
            data_loader_type,
            "--repeat",
            str(repeat),
            "--num_fetch_workers",
            str(1),
            "--limit",
            str(10),
            "--output_base_folder",
            str(output_base_folder),
        )
        results_files = glob.glob(f"{output_base_folder}/**/results-*.log", recursive=True)

        assert len(results_files) == num_workers + 1, results_files
        for f in results_files:
            parse_results_log(f)
        metadata_files = glob.glob(f"{output_base_folder}/**/*.json", recursive=True)
        assert len(metadata_files) == 2, metadata_files
        for f in metadata_files:
            with open(f, "r") as f:
                json.load(f)


@pytest.mark.parametrize(
    "action", ["random_gpu", "random_to_gpu", "single_image", "random_image", "mp"],
)
@pytest.mark.skipif(torch.cuda.is_available() is False, reason="Cuda device required.")
def test_benchmark_local_image_dataset(action: str):
    from benchmark.benchmark_tensor_loading import main

    with tempfile.TemporaryDirectory() as temp_dir:
        output_base_folder = Path(temp_dir) / "benchmark_results"
        output_base_folder.mkdir(exist_ok=False, parents=True)

        main("-a", action, "--output_base_folder", str(output_base_folder))
        results_files = glob.glob(f"{output_base_folder}/**/results*.log", recursive=True)
        assert len(results_files) == 1, results_files
        for f in results_files:
            parse_results_log(f)
