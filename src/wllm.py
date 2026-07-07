import argparse
import configparser
import sys

import experiment


class GreenListSizeRange(object):
    """Representing a range for float types between 0.0 and 1.0 (including)."""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __repr__(self):
        return "{0}-{1}".format(self.start, self.end)


def _read_config(args: argparse.Namespace, sys_argv: list[str]) -> argparse.Namespace:
    config = configparser.ConfigParser()
    config.read(args.config_file)
    argv = sys_argv
    for key, val in config["Pipeline"].items():
        arg = key.replace("_", "-")
        if arg[0] != "-":
            arg = f"--{arg}"
        argv.append(f"{arg}={val}")
    return parser.parse_args(argv)


parser = argparse.ArgumentParser(description="Watermarking LLM generated text")
parser.add_argument(
    "experiment",
    type=str,
    choices=["compare", "compute"],
    nargs=1,
    help="The type of experiment to run",
)
parser.add_argument(
    "-v",
    "--verbosity",
    type=int,
    default=0,
    choices=range(0, 3),
    help="Verbosity level of stdout output. Ranges from 0, i.e., low verbosity to 2.",
)
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="cpu",
    choices=["cpu", "cuda"],
    help="Device used for expensive computations, mainly matrix multiplications.",
)
parser.add_argument(
    "-s",
    "--sample-size",
    type=int,
    default=1,
    help="Number of samples to be drawn from the data source.",
)
parser.add_argument(
    "--random-sample",
    action=argparse.BooleanOptionalAction,
    help="True: sample randomly from the input data; False: Iterate over data in order.",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    choices=[
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
    ],
    default="facebook/opt-125m",
    help="Model weights used to generate watermarked text.",
)
parser.add_argument(
    "--oracle-model",
    type=str,
    choices=[
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
    ],
    default="facebook/opt-350m",
    help="Oracle model weights used to compute perplexity of the watermarked text.",
)
parser.add_argument(
    "--in-data",
    type=str,
    default="../data/realnews_tiny.jsonl",
    help="Path to the dataset used for generating prompts and baseline completions. Format needs to be jsonl.",
)
parser.add_argument(
    "--out-data",
    type=str,
    default="../data/archive/archive.jsonl",
    help="Path to the directory for archiving generated output and statistics.",
)
parser.add_argument(
    "--green-list-size",
    type=float,
    choices=[GreenListSizeRange(0.0, 1.0)],
    default=0.5,
    help="Size of the green list as a ratio of the tokenizers vocabulary.",
)
parser.add_argument(
    "--green-list-bias",
    type=float,
    default=2.0,
    help="Delta value added to all green list token before applying softmax.",
)
parser.add_argument(
    "--decoding-strategy",
    type=str,
    choices=["multinomial", "greedy", "beam"],
    default="greedy",
    help="Strategy for sampling token from the vocabulary given the models logit vector",
)
parser.add_argument(
    "--completion-size",
    type=int,
    default=200,
    help="Size of the input text that will be cut off from the end, to be used as a baseline completion.",
)
parser.add_argument(
    "--num-beams",
    type=int,
    default=4,
    help="Number of beams; Relevant only if decoding_strategy = beam.",
)
parser.add_argument(
    "--num-token-preview",
    type=int,
    default=75,
    help="Number of token to print when verbosity > 0",
)
parser.add_argument(
    "--from-config",
    action=argparse.BooleanOptionalAction,
    help="True: Read configuration from a configuration file; False: Read configuration from command line arguments.",
)
parser.add_argument(
    "--config-file",
    type=str,
    default="config.ini",
    help="Path to configuration file. This is only considered if --from-config is set.",
)


def main(args) -> None:
    if args.experiment[0] == "compare":
        experiment.compare(args)
    if args.experiment[0] == "compute":
        experiment.compute(args)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.from_config:
        args = _read_config(args, sys.argv[1:])
        print(f"Config loaded from file: {args}")

    main(args)
