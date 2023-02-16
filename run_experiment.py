import argparse
import json
import os
import pathlib
import sys
from collections import OrderedDict
from shutil import copyfile
from itertools import product

from my_library.model import Model
from my_library.trainer import Trainer
from my_library.utils import Benchmark, create_summary_table


DATASET_TO_METRIC = {
    "emotion": "accuracy"
}

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="paraphrase-mpnet-base-v2")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--allow_skip", default=True, type=bool)
    parser.add_argument("--sample_size", type=int, default=8)

    return parser


def create_results_path(dataset: str, split_name: str, output_path: str):
    results_path = os.path.join(output_path, dataset, split_name, "results.json")
    print(f"\n\n======== {os.path.dirname(results_path)} =======")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    return results_path


def setup(args_override):
    parser = get_argparser()
    if args_override is not None:
        args = parser.parse_args(args_override)
    else:
        args = parser.parse_args()

    print(f"Arguments: \n{args}")

    full_exp_name = f"{args.exp_name}".rstrip("-")
    parent_directory = pathlib.Path(__file__).parent.absolute()
    output_path = parent_directory / "results" / args.output_prefix / full_exp_name
    os.makedirs(output_path, exist_ok=True)

    with open(output_path / "hparams.json", "w") as f:
        f.write(json.dumps(vars(args), indent=2))

    # Save a copy of this training script and the run command in results directory
    train_script_path = os.path.join(output_path, "train_script.py")
    copyfile(__file__, train_script_path)
    with open(train_script_path, "a") as f_out:
        f_out.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))
    
    return args, output_path


def main(args_override=None):
    args, output_path = setup(args_override)

    for seed in range(args.num_seeds):
        for dataset in args.datasets:
            results_path = create_results_path(dataset, f"train-{args.sample_size}-{seed}", output_path)

            if os.path.exists(results_path) and args.allow_skip:
                print(f"Skipping finished experiment: {results_path}")
                exit()

            metric = DATASET_TO_METRIC.get(dataset, "accuracy")
            print(f"Evaluating {dataset} using {metric!r}.")

            # Load model
            model = Model.from_pretrained(args.model)

            # Train on augmented data only
            trainer = Trainer(
                model=model,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                metric=metric
            )
            trainer.train()

            # Evaluate the model on the test data
            metrics = trainer.evaluate()
            print(f"Metrics: {metrics}")

            with open(results_path, "w") as f:
                result = {"score": metrics[metric] * 100, "measure": metric}
                json.dump(result, f, sort_keys=True)

    # Create a summary_table.csv file that computes means and standard deviations
    # for all of the results in `output_path`.
    average_score = create_summary_table(str(output_path))
    return average_score


def run_hparam_search():
    hparams_space = OrderedDict(
        sample_size = [4, 8]
    )

    parent_directory = pathlib.Path(__file__).parent.absolute()
    output_path = parent_directory / "results" / "hparam_sweep"

    # Prepare experiments list
    hparam_space = []
    for hparam_values in product(*hparams_space.values()):
        hparam_dict = dict(zip(hparams_space.keys(), hparam_values))

        # Filter illegal hparam combinations
        if (hparam_dict["continual"] and hparam_dict["select_iterations"] == 1) or \
            hparam_dict["top_range_start"] >= hparam_dict["top_range_end"] or \
            hparam_dict["bottom_range_start"] >= hparam_dict["bottom_range_end"]:
            continue
        hparam_space.append(hparam_dict)
        
    # Run experiments
    bench = Benchmark()
    max_average_score = -1
    for i, hparam_set in enumerate(hparam_space):
        with bench.track(f"Experiment_{i}"):
            # Experiment name and constant arguments
            args = ["--exp_name", f"experiment_{i}", "--output_prefix", "hparam_sweep"]

            # Prepare argument list
            for arg_name, arg_value in hparam_set.items():
                if type(arg_value) is bool:
                    arg_value = "True" if arg_value else ""
                args.extend([f"--{arg_name}", str(arg_value)])

            average_score = main(args)

            if average_score > max_average_score:
                max_average_score = average_score
                with open(output_path / "best_score.txt", "w") as f:
                    f.write(f"score: {max_average_score}\nExperiment no: {i}\n\n")
                    f.write(f"hparams set:\n\n{json.dumps(hparam_dict, indent=2)}")


if __name__ == "__main__":
    if sys.argv[1:]:
        main()

    # If no arguments, run hparam search
    else:
        print("there")
        run_hparam_search()
