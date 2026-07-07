import os
import torch
import pandas as pd

from typing import Tuple
from transformers import (
    GPT2Tokenizer,
    OPTForCausalLM,
    LogitsProcessorList,
    GenerationConfig,
)


class Data:
    def __init__(self, sample_size: int, random_sample: bool, in_path: str) -> None:
        self.sample_size = sample_size
        self.random_sample = random_sample
        self.index = 0
        data = pd.read_json(path_or_buf=in_path, lines=True)
        self.text = data.get("text")

    def __iter__(self):
        if self.random_sample:
            return self
        else:
            list = self.text.tolist()[: self.sample_size]
            return iter(list)

    def __next__(self):
        if self.index < self.sample_size:
            self.index += 1
            return self.text.sample(n=1).item()
        else:
            raise StopIteration


class Pipeline:
    def __init__(
        self,
        args,
        rng: torch.Generator = torch.Generator(),
        ppl_stride: int = 512,
    ):
        self.weights = args.model
        self.oracle_weights = args.oracle_model
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.weights)
        self.oracle_tokenizer = GPT2Tokenizer.from_pretrained(self.oracle_weights)

		# Reduce CPU mem usage: https://huggingface.co/docs/transformers/main_classes/model#large-model-loading
		#
		# low_cpu_mem_usage=True 	loads pretrained weights in-place reducing mem requirements to the size of the model
		# device_map="auto"		places the model on different devices, also sets low_cpu_mem_usage=True
		# torch_dtype="auto"		possibly reduces floating point accuracy
        self.model = OPTForCausalLM.from_pretrained(self.weights, device_map=0, torch_dtype=torch.bfloat16)
        self.oracle_model = OPTForCausalLM.from_pretrained(self.oracle_weights, device_map=1, torch_dtype=torch.bfloat16)
        self.dataset_path = args.in_data
        self.dataset_sample_size = args.sample_size
        self.completion_size = args.completion_size
        self.max_new_tokens = self.completion_size + 5
        self.min_new_tokens = self.completion_size - 5
        self.vocab = list(self.tokenizer.get_vocab().values())
        self.green_list_ratio = args.green_list_size
        self.green_list_bias = args.green_list_bias
        self.rng = rng
        self.ppl_stride = ppl_stride
        self.device = args.device
        self.decoding_strategy = args.decoding_strategy
        self.num_beams = args.num_beams

    def create_prompt(self, sample: str) -> Tuple[str, str, str, str]:
        """Create prompt and basecompletion from dataset."""

        # sample = longprompt.get(self.dataset_path)
        # print(f"sample: {len(sample)}")

        # Truncate prompt to max sequence length
        prompt_max_len = self.model.config.max_position_embeddings
        token = self.tokenizer(
            sample, add_special_tokens=True, truncation=True, max_length=prompt_max_len
        )["input_ids"]
        prompt = token[: -self.max_new_tokens]

        decoded_prompt = self.tokenizer.decode(prompt, skip_special_tokens=True)

        base_completion = token[-self.max_new_tokens :]

        decoded_base_completion = self.tokenizer.decode(
            base_completion, skip_special_tokens=True
        )
        return prompt, base_completion, decoded_prompt, decoded_base_completion

    def generate(self, prompt, watermark_processor, watermark=False) -> str:
        """
        Generation
        Greedy search - num_beams=1, do_sample=False
        Multinomial sampling - num_beamns=1, do_sample=True
        Beam search - num_beams=n, do_sample=False

        Logits warping - soft rule: sequence_bias, hard rule: suppress_tokens
        https://huggingface.co/docs/transformers/v4.35.0/en/internal/generation_utils
        """
        config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.eos_token_id,
            # remove_invalid_values=True
            # renormalize_logits=True
            # might be needed when applying logits processors
        )
        if self.decoding_strategy == "greedy":
            config.num_beams = 1
            config.do_sample = False
        elif self.decoding_strategy == "multinomial":
            config.do_sample = True
        elif self.decoding_strategy == "beam":
            config.num_beams = self.num_beams
            config.do_sample = False

        model = self.model#.to(self.device)
        token_tensor = torch.tensor(prompt).unsqueeze(0).to("cuda:0")
        if not watermark:
            out_sequence = model.generate(token_tensor, config)[0]
            text = self.tokenizer.decode(
                out_sequence,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            return text
        else:
            watermarked_out_sequence = model.generate(
                token_tensor,
                config,
                logits_processor=LogitsProcessorList([watermark_processor]),
            )[0]
            watermarked_text = self.tokenizer.decode(
                watermarked_out_sequence,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            return watermarked_text

    def compute_ppl(self, text, prompt_len) -> float:
        """
        Computing Perplexity PPL, Code ported from https://huggingface.co/docs/transformers/perplexity
        """
        oracle_model = self.oracle_model#.to(self.device)
        max_length = oracle_model.config.max_position_embeddings
        out_encoding = self.oracle_tokenizer(text[prompt_len:], return_tensors="pt")
        out_encoding = out_encoding.to("cuda:1")
        seq_len = out_encoding.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.ppl_stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            ids = out_encoding.input_ids[:, begin_loc:end_loc]
            target_ids = ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = oracle_model(ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        return ppl.item()
