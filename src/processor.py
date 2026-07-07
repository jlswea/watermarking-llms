from dataclasses import dataclass, field
from math import sqrt
from typing import Tuple

import numpy as np
import torch
from transformers import GPT2Tokenizer, LogitsProcessor


@dataclass
class Result:
    scores: [float] = field(default_factory=lambda: [np.nan])
    final_z: float = np.nan
    tagged_text: str = np.nan
    num_generated: int = np.nan


class WatermarkProcessor:
    def __init__(
        self, args, vocab: list[int] = None, rng: torch.Generator = None
    ) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.green_list_ratio = args.green_list_size
        self.green_list_size = int(self.vocab_size * self.green_list_ratio)
        self.green_list_bias = args.green_list_bias
        self.device = args.device
        self.rng = rng

        assert vocab is not None
        assert rng is not None

    def _compute_z(self, num_generated: int, num_green_list_token: int):
        if num_generated == 0:
            return 0
        else:
            num_expected_green_token = self.green_list_ratio * num_generated
            delta = num_green_list_token - num_expected_green_token
            z = delta / sqrt(num_expected_green_token * (1 - self.green_list_ratio))
            return z


class WatermarkLogitsProcessor(WatermarkProcessor, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Seed RNG
        token = input_ids[0][-1].item()
        self.rng.manual_seed(token)  # Potentially have to use larger seed

        # Create green list
        perm_vocab = torch.randperm(
            self.vocab_size + 1, generator=self.rng, device=self.device
        )  # Compute on CPU?
        green_list_vocab = perm_vocab[: self.green_list_size]

        # Apply bias
        for green_list_id in green_list_vocab:  # Use a mask instead?
            scores[0][green_list_id] = scores[0][green_list_id] + self.green_list_bias

        return scores


class WatermarkDetector(WatermarkProcessor):
    def __init__(self, *args, tokenizer: GPT2Tokenizer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

        assert tokenizer is not None

    def detect_watermark(self, watermarked_text, len_prompt) -> Tuple[float, int, str]:
        """Detect the watermark, return z score"""
        num_green_list_token = 0
        num_red_list_token = 0
        tagged_text = ""
        scores = []
        out_tokens = self.tokenizer(watermarked_text)["input_ids"]
        for idx, out_token in enumerate(out_tokens):
            if idx == 0 and out_token == 2:
                continue  # Manually skip BOS token
            tagged_text += self.tokenizer.decode(
                out_token, clean_up_tokenization_spaces=True, skip_special_tokens=True
            )
            if idx < len_prompt:
                continue
            self.rng.manual_seed(out_tokens[idx - 1])  # Seed rng with prev token
            green_list_size = int(len(self.vocab) * self.green_list_ratio)
            perm_vocab = torch.randperm(
                len(self.vocab) + 1, generator=self.rng, device=self.device
            )  # Compute on CPU?
            green_list_vocab = perm_vocab[:green_list_size]

            if out_token in green_list_vocab:
                num_green_list_token += 1
                tagged_text = tagged_text + "{g}"

            else:
                num_red_list_token += 1
                tagged_text = tagged_text + "{r}"

            scores.append(
                self._compute_z(
                    num_generated=(idx - len_prompt),
                    num_green_list_token=num_green_list_token,
                )
            )

        # Compute final z score
        num_generated = (
            len(out_tokens) - len_prompt
        )  # First token cannot be watermarked; Don't consider prompt
        return Result(
            scores=scores,
            final_z=self._compute_z(
                num_generated=num_generated, num_green_list_token=num_green_list_token
            ),
            num_generated=num_generated,
            tagged_text=tagged_text,
        )
