import pandas as pd
import torch
from tabulate import tabulate

import archive
from pipeline import Data, Pipeline
from processor import WatermarkDetector, WatermarkLogitsProcessor


## Utils
def _format(sequence, len_prompt, num_token_to_print=50):
    formatted = sequence[len_prompt : len_prompt + num_token_to_print]
    return formatted.replace("\n", " ")


def compare(args) -> None:
    num_token_to_print = args.num_token_preview
    pipeline = Pipeline(args, rng=torch.Generator(device=args.device))
    watermark_processor = WatermarkLogitsProcessor(
        args,
        vocab=pipeline.vocab,
        rng=torch.Generator(device=args.device),
    )
    watermark_detector = WatermarkDetector(
        args,
        tokenizer=pipeline.tokenizer,
        vocab=pipeline.vocab,
        rng=torch.Generator(device=args.device),
    )

    data = Data(args.sample_size, args.random_sample, in_path=args.in_data)

    archive_data = []

    for idx, sample in enumerate(data):
        if sample == "":
            continue

        if args.verbosity > 0:
            print(120 * "=")
            print(f"Sample {idx+1}: ")
            print(" ")

        # Prepare data
        archive_datum = archive.Archive()

        (
            prompt,
            base_completion,
            decoded_prompt,
            decoded_base_completion,
        ) = pipeline.create_prompt(sample)
        if len(prompt) == 0:
            continue
        archive_datum.len_prompt = len(prompt)

        if args.verbosity > 1:
            end_decoded_prompt = decoded_prompt[-num_token_to_print:].replace("\n", " ")
            print(f"Prompt ends with: ... {end_decoded_prompt}")
            start_decoded_base_completion = decoded_base_completion[
                :num_token_to_print
            ].replace("\n", " ")
            print(
                f"Baseline completion starts with:        {start_decoded_base_completion} ..."
            )
            print(120 * "-")

        # Generate and test LLM without watermark
        text = pipeline.generate(prompt, None)
        res = watermark_detector.detect_watermark(text, len(prompt))
        archive_datum.z_score = res.final_z
        archive_datum.len_completion = res.num_generated
        ppl = pipeline.compute_ppl(text, len(prompt))
        archive_datum.ppl = ppl

        if args.verbosity > 1:
            start_text = _format(
                text,
                len_prompt=len(decoded_prompt),
                num_token_to_print=num_token_to_print,
            )
            print(f"Generated completation starts with:     {start_text} ...")
            start_tagged_text = _format(
                res.tagged_text,
                len_prompt=len(decoded_prompt),
                num_token_to_print=num_token_to_print,
            )
            print(f"Tagged completion starts with:          {start_tagged_text} ...")
            print(120 * "-")

        # Generate and test watermarked text
        watermarked_text = pipeline.generate(
            prompt, watermark_processor, watermark=True
        )
        if args.verbosity > 1:
            start_wartermarked_text = _format(
                watermarked_text,
                len_prompt=len(decoded_prompt),
                num_token_to_print=num_token_to_print,
            )
            print(
                f"Watermarked completation starts with:   {start_wartermarked_text} ..."
            )

        watermark_res = watermark_detector.detect_watermark(
            watermarked_text, len(prompt)
        )
        archive_datum.watermarked_z_score = watermark_res.final_z
        archive_datum.len_watermarked_completion = watermark_res.num_generated
        watermarked_ppl = pipeline.compute_ppl(watermarked_text, len(prompt))
        archive_datum.watermarked_ppl = watermarked_ppl
        if args.verbosity > 1:
            start_watermarked_tagged_text = _format(
                watermark_res.tagged_text,
                len_prompt=len(decoded_prompt),
                num_token_to_print=num_token_to_print,
            )
            print(
                f"Watermarked, tagged text starts with:   {start_watermarked_tagged_text} ..."
            )
            print(120 * "-")

        # Archive data
        archive_datum.prompt = decoded_prompt
        archive_datum.baseline_completion = decoded_base_completion
        archive_datum.generated_completion = text
        archive_datum.generated_tagged_completion = res.tagged_text
        archive_datum.watermarked_completion = watermarked_text
        archive_datum.watermarked_tagged_completion = watermark_res.tagged_text
        archive_datum.scores = res.scores
        archive_datum.watermarked_scores = watermark_res.scores

        archive_data.append(archive_datum)

        if args.verbosity > 0:
            print(
                tabulate(
                    [
                        ["Yes", watermark_res.final_z, watermarked_ppl],
                        ["No", res.final_z, ppl],
                    ],
                    headers=["Watermark", "Z score", "PPL"],
                )
            )

    archive.write(archive_data, args.out_data)
    watermarked_z_scores = [d.watermarked_z_score for d in archive_data]
    watermarked_ppls = [d.watermarked_ppl for d in archive_data]
    z_scores = [d.z_score for d in archive_data]
    ppls = [d.ppl for d in archive_data]

    print(60 * "=")
    print(f"Summary of {args.sample_size} samples: \n")
    print(
        tabulate(
            [
                [
                    "Yes",
                    torch.tensor(watermarked_z_scores).mean(),
                    torch.tensor(watermarked_ppls).mean(),
                ],
                ["No", torch.tensor(z_scores).mean(), torch.tensor(ppls).mean()],
            ],
            headers=["Watermark", "Mean Z score", "Mean PPL"],
        )
    )


def compute(args) -> None:
    pipeline = Pipeline(args, rng=torch.Generator(device=args.device))
    watermark_processor = WatermarkLogitsProcessor(
        args,
        vocab=pipeline.vocab,
        rng=torch.Generator(device=args.device),
    )
    watermark_detector = WatermarkDetector(
        args,
        tokenizer=pipeline.tokenizer,
        vocab=pipeline.vocab,
        rng=torch.Generator(device=args.device),
    )
    data = Data(args.sample_size, args.random_sample, in_path=args.in_data)

    watermarked_z_scores = []
    watermarked_ppls = []

    for idx, sample in enumerate(data):
        if sample == "":
            continue

        # Prepare data
        (
            prompt,
            _,
            decoded_prompt,
            _,
        ) = pipeline.create_prompt(sample)
        if len(prompt) == 0:
            continue

        preview = decoded_prompt[:50].replace("\n", " ")
        print(f"{idx} - {preview} ...")
        watermarked_text = pipeline.generate(
            prompt, watermark_processor, watermark=True
        )

        res = watermark_detector.detect_watermark(watermarked_text, len(prompt))
        watermarked_z_scores.append(res.final_z)
        watermarked_ppl = pipeline.compute_ppl(watermarked_text, len(prompt))
        watermarked_ppls.append(watermarked_ppl)

    print(60 * "=")
    print(f"Summary of {args.sample_size} samples: \n")
    print(
        tabulate(
            [
                ["Mean Z score", torch.tensor(watermarked_z_scores).mean()],
                ["Mean PPL", torch.tensor(watermarked_ppls).mean()],
            ],
        )
    )
