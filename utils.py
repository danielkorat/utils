from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import losses

from .data import create_fewshot_splits, create_fewshot_splits_multilabel
from .modeling import SupConLoss

import argparse
import json
import os
import tarfile
from collections import defaultdict
from glob import glob
from os import listdir
from os.path import isdir, join, splitext
from typing import List, Tuple

from numpy import mean, median, std
from scipy.stats import iqr


SEC_TO_NS_SCALE = 1000000000


def extract_results(path: str) -> None:
    tar = tarfile.open(path, "r:gz")
    unzip_path = splitext(splitext(path)[-2])[-2]
    tar.extractall(path=os.path.dirname(unzip_path))
    tar.close()
    return unzip_path


def get_sample_sizes(path: str) -> List[str]:
    return sorted(list({int(name.split("-")[-2]) for name in glob(f"{path}/*/train-*-0")}))


def get_formatted_ds_metrics(path: str, dataset: str, sample_sizes: List[str]) -> Tuple[str, List[str]]:
    formatted_row = []
    metric_name = ""
    exact_metrics, exact_stds = {}, {}

    for sample_size in sample_sizes:
        result_jsons = sorted(glob(os.path.join(path, dataset, f"train-{sample_size}-*", "results.json")))
        split_metrics = []

        for result_json in result_jsons:
            with open(result_json) as f:
                result_dict = json.load(f)

            metric_name = result_dict.get("measure", "N/A")
            split_metrics.append(result_dict["score"])

        exact_metrics[sample_size] = mean(split_metrics)
        exact_stds[sample_size] = std(split_metrics)
        formatted_row.extend([f"{exact_metrics[sample_size]:.1f}", f"{exact_stds[sample_size]:.1f}"])

    return metric_name, formatted_row, exact_metrics, exact_stds, sample_sizes


def create_summary_table(results_path: str) -> None:
    """Given per-split results, creates a summary table of all datasets,
    with average metrics and standard deviations.

    Args:
        path: path to per-split results: either `scripts/{method_name}/{results}/{model_name}`,
            or `final_results/{method_name}/{model_name}.tar.gz`
    """

    if results_path.endswith("tar.gz"):
        unzipped_path = extract_results(results_path)
    else:
        unzipped_path = results_path

    sample_sizes = get_sample_sizes(unzipped_path)
    header_row = ["dataset", "measure"]
    for sample_size in sample_sizes:
        header_row.append(f"{sample_size}_avg")
        header_row.append(f"{sample_size}_std")

    csv_lines = [header_row]

    means, stds = defaultdict(list), defaultdict(list)
    for dataset in next(os.walk(unzipped_path))[1]:
        metric_name, formatted_metrics, exact_metrics, exact_stds, sample_sizes = get_formatted_ds_metrics(
            unzipped_path, dataset, sample_sizes
        )
        dataset_row = [dataset, metric_name, *formatted_metrics]
        csv_lines.append(dataset_row)

        # Collect exact metrics for overall average and std calculation
        for sample_size in sample_sizes:
            means[sample_size].append(exact_metrics[sample_size])
            stds[sample_size].append(exact_stds[sample_size])

    # Generate row for overall average
    formatted_average_row = []
    for sample_size in sample_sizes:
        overall_average = mean(means[sample_size])
        overall_std = mean(stds[sample_size])
        formatted_average_row.extend([f"{overall_average:.1f}", f"{overall_std:.1f}"])
    csv_lines.append(["Average", "N/A", *formatted_average_row])

    output_path = os.path.join(unzipped_path, "summary_table.csv")
    print("=" * 80)
    print("Summary table:\n")
    with open(output_path, "w") as f:
        for line in csv_lines:
            f.write(",".join(line) + "\n")
            print(", ".join(line))
    print("=" * 80)
    print(f"Saved summary table to {output_path}")
    return overall_average


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    create_summary_table(args.path)


@dataclass
class Benchmark:
    """
    Performs simple benchmarks of code portions (measures elapsed time).

        Typical usage example:

        bench = Benchmark()
        with bench.track("Foo function"):
            foo()
        with bench.track("Bar function"):
            bar()
        bench.summary()
    """

    out_path: Optional[str] = None
    summary_msg: str = field(default_factory=str)

    def print(self, msg: str) -> None:
        """
        Prints to system out and optionally to specified out_path.
        """
        print(msg)

        if self.out_path is not None:
            with open(self.out_path, "a+") as f:
                f.write(msg + "\n")

    @contextmanager
    def track(self, step):
        """
        Computes the elapsed time for given code context.
        """
        start = monotonic_ns()
        yield
        ns = monotonic_ns() - start
        msg = f"\n{'*' * 70}\n'{step}' took {ns / SEC_TO_NS_SCALE:.3f}s ({ns:,}ns)\n{'*' * 70}\n"
        print(msg)
        self.summary_msg += msg + "\n"

    def summary(self) -> None:
        """
        Prints summary of all benchmarks performed.
        """
        self.print(f"\n{'#' * 30}\nBenchmark Summary:\n{'#' * 30}\n\n{self.summary_msg}")
        
