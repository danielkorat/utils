from contextlib import contextmanager
from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import losses

from .data import create_fewshot_splits, create_fewshot_splits_multilabel
from .modeling import SupConLoss


SEC_TO_NS_SCALE = 1000000000


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
