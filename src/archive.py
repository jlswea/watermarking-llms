import argparse
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

import numpy as np
import pandas as pd


class Mode(str, Enum):
    read = "read"
    get_results = "get_results"
    get_scores = "get_scores"
    get_confidence = "get_confidence"
    get_samples = "get_samples"


@dataclass
class Archive:
    prompt: str = np.nan
    baseline_completion: str = np.nan
    generated_completion: str = np.nan
    generated_tagged_completion: str = np.nan
    watermarked_completion: str = np.nan
    watermarked_tagged_completion: str = np.nan
    z_score: float = np.nan
    scores: [float] = field(default_factory=lambda: [np.nan])
    ppl: float = np.nan
    watermarked_z_score: float = np.nan
    watermarked_scores: [float] = field(default_factory=lambda: [np.nan])
    watermarked_ppl: float = np.nan
    len_watermarked_completion: int = np.nan
    len_completion: int = np.nan
    len_prompt: int = np.nan


parser = argparse.ArgumentParser(
    description="Parsing an archived wllm-pipeline.py output"
)

parser.add_argument(
    "mode",
    type=str,
    choices=[
        Mode.read,
        Mode.get_results,
        Mode.get_scores,
        Mode.get_confidence,
        Mode.get_samples,
    ],
)
parser.add_argument("path", type=str, help="Path to the input file")
parser.add_argument("--filter", type=str, help="Substring to filter path names")
parser.add_argument("--threshold", type=int, help="Threshold for the classification")
parser.add_argument(
    "--num_token", type=int, help="Number of token to be considered", default=200
)


def _tail(path: str, n: int = 1):
    """
    Returns the n last lines of a text file.
    Generalized from SO: https://stackoverflow.com/a/18603065
    """
    with open(path, "rb") as f:
        lines = 0
        try:
            f.seek(-2, os.SEEK_END)  # Jump to the second last byte.
            while lines < n:
                while f.read(1) != b"\n":  #  Until newline is found ...
                    f.seek(
                        -2, os.SEEK_CUR
                    )  #  ... jump back, over the read byte plus one.
                lines += 1
                f.seek(-2, os.SEEK_CUR)
            f.seek(2, os.SEEK_CUR)
        except OSError:  # Reached begginning of File
            f.seek(0)  #  Set cursor to beginning of file as well.
        return f.read().decode()  # Read all data from this point on.


def _item(keyval: str) -> Tuple[str, str]:
    """Splits a key-value naming convention into a (key, value) tuple"""
    ds = re.search(r"\d", keyval)
    if ds:
        d = ds.start()
        return keyval[:d], keyval[d:]
    else:
        raise ValueError("No value found")


def read(args: argparse.Namespace) -> None:
    json = pd.read_json(path_or_buf=args.path, lines=True)
    print(json)


def get_scores(args: argparse.Namespace) -> None:
    dirs = [d for d in os.scandir(args.path) if d.is_dir()]
    if args.filter:
        dirs = [d for d in dirs if args.filter in d.path.split("/")[2]]

    for d in dirs:
        print(d.path)
        try:
            json = pd.read_json(
                path_or_buf=os.path.join(d, "archive.jsonl"), lines=True
            )
        except:
            print("No archive to read")
            continue

        # scores = pd.DataFrame.from_records(json.get("scores"))
        wm_scores = pd.DataFrame.from_records(json.get("watermarked_scores"))

        # print(scores.mean())
        res = wm_scores.mean()[:200]
        print(res)
        res.to_csv(path_or_buf=os.path.join(d, "scores.csv"), header=False)


def get_confidence(args: argparse.Namespace) -> None:
    if not args.threshold:
        print("No threshold")
        return
    dirs = [d for d in os.scandir(args.path) if d.is_dir()]
    print(dirs)
    if args.filter:
        dirs = [d for d in dirs if args.filter in d.path.split("/")[2]]

    for d in dirs:
        print(d.path)
        try:
            json = pd.read_json(
                path_or_buf=os.path.join(d, "archive.jsonl"), lines=True
            )
        except:
            print("No archive to read")
            continue

        scores = pd.DataFrame.from_records(json.get("scores")).mean()[:200]
        wm_scores = pd.DataFrame.from_records(json.get("watermarked_scores")).mean()[
            :200
        ]

        fpr = []
        tpr = []
        fnr = []
        tnr = []

        fp = 0
        tp = 0
        fn = 0
        tn = 0

        for z in wm_scores:
            if z > args.threshold:
                tp += 1
            else:
                fn += 1
            fnr.append(0 if (fn + tp) <= 0 else fn / (fn + tp))
            tpr.append(0 if (tp + fn) <= 0 else tp / (tp + fn))

        for z in scores:
            if z > args.threshold:
                fp += 1
            else:
                tn += 1
            fpr.append(0 if (fp + tn) <= 0 else fp / (fp + tn))
            tnr.append(0 if (tn + fp) <= 0 else tn / (tn + fp))

        res = pd.DataFrame(data={"FPR": fpr, "TNR": tnr, "TPR": tpr, "FNR": fnr})

        print(res)

        res.to_csv(path_or_buf=os.path.join(d, f"confidence-t{args.threshold}.csv"))


def get_samples(args: argparse.Namespace) -> None:
    if not args.threshold:
        print("No threshold")
        return
    dirs = [d for d in os.scandir(args.path) if d.is_dir()]
    if args.filter:
        dirs = [d for d in dirs if args.filter in d.path.split("/")[2]]

    for d in dirs:
        print(d.path)
        try:
            json = pd.read_json(
                path_or_buf=os.path.join(d, "archive.jsonl"), lines=True
            )
        except:
            print("No archive to read")
            continue

        fn = []
        tp = []
        tn = []
        fp = []

        samples = pd.DataFrame.from_records(json)
        for idx, row in samples.iterrows():
            if row["watermarked_scores"][args.num_token - 1] <= args.threshold:
                fn.append(row)
            elif row["watermarked_scores"][args.num_token - 1] > args.threshold:
                tp.append(row)

            if row["scores"][args.num_token - 1] <= args.threshold:
                tn.append(row)
            elif row["scores"][args.num_token - 1] > args.threshold:
                fp.append(row)

        fn_df = pd.DataFrame(fn)
        tp_df = pd.DataFrame(tp)
        tn_df = pd.DataFrame(tn)
        fp_df = pd.DataFrame(fp)

        res = {"type": [], "prompt": [], "completion": [], "ppl": [], "zscore": []}

        fn_sample = fn_df.sample().iloc[0] if 1 <= fn_df.shape[0] else 0
        res["type"].append("FN")
        res["prompt"].append(fn_sample["prompt"])
        res["completion"].append(
            fn_sample["watermarked_tagged_completion"][len(fn_sample["prompt"]) :]
        )
        res["ppl"].append(fn_sample["watermarked_ppl"])
        res["zscore"].append(fn_sample["watermarked_scores"][args.num_token - 1])

        tp_sample = tp_df.sample().iloc[0] if 1 <= tp_df.shape[0] else 0
        res["type"].append("TP")
        res["prompt"].append(tp_sample["prompt"])
        res["completion"].append(
            tp_sample["watermarked_tagged_completion"][len(tp_sample["prompt"]) :]
        )
        res["ppl"].append(tp_sample["watermarked_ppl"])
        res["zscore"].append(tp_sample["watermarked_scores"][args.num_token - 1])

        tn_sample = tn_df.sample().iloc[0] if 1 <= tn_df.shape[0] else 0
        res["type"].append("TN")
        res["prompt"].append(tn_sample["prompt"])
        res["completion"].append(
            tn_sample["generated_tagged_completion"][len(tn_sample["prompt"]) :]
        )
        res["ppl"].append(tn_sample["ppl"])
        res["zscore"].append(tn_sample["scores"][args.num_token - 1])

        fp_sample = fp_df.sample().iloc[0] if 1 <= fp_df.shape[0] else 0
        if fp_sample != 0:
            res["type"].append("FP")
            res["prompt"].append(fp_sample["prompt"])
            res["completion"].append(
                fp_sample["generated_tagged_completion"][len(fp_sample["prompt"]) :]
            )
            res["ppl"].append(fp_sample["ppl"])
            res["zscore"].append(fp_sample["scores"][args.num_token - 1])

        res_df = pd.DataFrame(data=res)
        print(res_df)
        res_df.to_csv(
            path_or_buf=os.path.join(
                d, f"samples-t{args.threshold}-n{args.num_token}.csv"
            )
        )


def get_results(args: argparse.Namespace) -> None:
    # Get stdout.txt path from all subdirs of args.path
    files = [
        os.path.join(f.path, "stdout.txt") for f in os.scandir(args.path) if f.is_dir()
    ]

    experiments = []
    gammas = []
    deltas = []
    watermarked_mean_zscores = []
    watermarked_mean_ppls = []
    mean_zscores = []
    mean_ppls = []

    for f in files:
        # Get last lines of stdout.txt
        try:
            tail = _tail(f, n=2)
        except:
            print(f"failed to open {f}")
            continue

        lines = tail.splitlines()
        array = []
        for line in lines:
            array.append(line.split())
        # Get results from table
        dir = os.path.dirname(f).split("/")[-1]
        dir_split = dir.split("-")
        experiments.append(f"{dir_split[0]}-{dir_split[3]}")
        gammas.append(_item(dir_split[1])[1])
        deltas.append(_item(dir_split[2])[1])
        watermarked_mean_zscores.append(array[0][1])
        watermarked_mean_ppls.append(array[0][2])
        mean_zscores.append(array[1][1])
        mean_ppls.append(array[1][2])

    res = {
        "experiment": experiments,
        "gamma": gammas,
        "delta": deltas,
        "watermarked_mean_zscores": watermarked_mean_zscores,
        "watermarked_mean_ppls": watermarked_mean_ppls,
        "mean_zscores": mean_zscores,
        "mean_ppl": mean_ppls,
    }

    archive_df = pd.DataFrame(data=res)
    archive_df.sort_values(["delta", "gamma"], inplace=True)
    res_path = os.path.join(args.path, "result.csv")
    archive_df.to_csv(path_or_buf=res_path, index=False)


def write(archive_data: [Archive], path: str) -> None:
    archive_df = pd.DataFrame(data=archive_data)
    archive_df.to_json(path_or_buf=path, orient="records", lines=True)
    print(archive_df)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == Mode.read:
        read(args)
    if args.mode == Mode.get_results:
        get_results(args)
    if args.mode == Mode.get_scores:
        get_scores(args)
    if args.mode == Mode.get_confidence:
        get_confidence(args)
    if args.mode == Mode.get_samples:
        get_samples(args)
