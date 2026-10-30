from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import transformers
from transformers import GenerationConfig

from piggen import utils

log_format = "%(asctime)s - %(levelname)s - %(message)s"
date_format = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
logger = logging.getLogger(__name__)


class pIgGen:

    def __init__(
        self,
        model_name: Literal[
            "ollieturnbull/p-IgGen", "ollieturnbull/p-IgGen-developable"
        ] = "ollieturnbull/p-IgGen",
        device=None,
        cache_dir=None,
    ):
        logger.info("Initializing p-IgGen model...")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir
        )

        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                # Check if MPS is both available and built
                self.device = "mps"
            else:
                self.device = "cpu"
            logger.info(f"No device specified, automatically selected {self.device}.")
        else:
            self.device = device
        self.model.to(self.device)
        logger.info(f"p-IgGen model initialized on {self.device}.")

    def _generate(
        self,
        num_return_sequences: int,
        prompt_sequence: None,
        eos_token_id: int,
        temp=1.25,
        top_p=0.95,
        batch_size=1,
    ):
        input_ids = self.tokenizer.encode(prompt_sequence, return_tensors="pt").to(
            self.device
        )
        pad_token_id = self.tokenizer.pad_token_id
        generation_config = GenerationConfig(
            do_sample=True,
            top_p=top_p,
            pad_token_id=pad_token_id,
            max_new_tokens=400,
            temperature=temp,
            num_return_sequences=num_return_sequences,
            eos_token_id=eos_token_id,
            device=self.device,
            batch_size=batch_size,
        )
        generated_token_ids = []

        generated_token_ids.append(
            self.model.generate(
                input_ids=input_ids, generation_config=generation_config
            )
        )

        # flatten the generated token ids
        generated_token_ids = [
            item for sublist in generated_token_ids for item in sublist
        ]

        decoded_sequences = self.tokenizer.batch_decode(
            generated_token_ids, skip_special_tokens=True
        )

        return decoded_sequences

    def generate(
        self,
        num_return_sequences: int,
        backwards=False,
        top_p=0.95,
        temp=1.25,
        batch_size=1,
        prompt: str | None = None,
        discard_bottom_n_percent=None,
        separated_output=False,
    ) -> list[str]:

        if backwards:
            if prompt is None:
                prompt = "2"
            eos_token_id = self.tokenizer.encode("1")[0]
        else:
            if prompt is None:
                prompt = "1"
            eos_token_id = self.tokenizer.encode("2")[0]

        if discard_bottom_n_percent is not None and num_return_sequences < 100:
            logger.warning(
                """
                Cannot discard bottom n percent with less than 100 sequences.
                Ignoring discard_bottom_n_percent.
                """
            )
            discard_bottom_n_percent = None

        if discard_bottom_n_percent is not None and num_return_sequences >= 100:
            n_samples = num_return_sequences
            num_return_sequences = math.ceil(
                num_return_sequences / (1 - discard_bottom_n_percent / 100)
            )

        generated_sequences = self._generate(
            num_return_sequences,
            prompt,
            eos_token_id=eos_token_id,
            temp=temp,
            top_p=top_p,
            batch_size=batch_size,
        )

        # remove eos and bos tokens, and reverse if backwards
        decoded_sequences = [
            utils.format_and_validate_output(sequence)
            for sequence in generated_sequences
        ]

        # remove any sequences that returned as None
        decoded_sequences = [
            sequence for sequence in decoded_sequences if sequence is not None
        ]

        if discard_bottom_n_percent is not None:
            likelihoods = self.get_batch_log_likelihoods(
                decoded_sequences, batch_size=batch_size
            )
            decoded_sequences = zip(decoded_sequences, likelihoods)
            decoded_sequences = sorted(
                decoded_sequences, key=lambda x: x[1], reverse=True
            )
            decoded_sequences = decoded_sequences[:n_samples]
            decoded_sequences = [x[0] for x in decoded_sequences[:n_samples]]

        logger.info(f"Generated {len(decoded_sequences)} sequences with temp {temp}.")
        if separated_output:
            VH, VL = utils.get_separate_VH_VL(decoded_sequences)
            return VH, VL
        return decoded_sequences

    def generate_heavy_chain(
        self,
        light_chain: str,
        num_return_sequences: int,
        top_p=0.95,
        temp=1.25,
        batch_size=1,
    ) -> list[str]:
        """
        Given a light chain sequence, generate a heavy chain sequence.
        Will genreate in reverse direction.

        Args:
            light_chain (str): The light chain sequence.
            num_return_sequences (int): The number of heavy chain sequences to generate.
            top_p (float, optional): The cumulative probability for nucleus sampling.
                Defaults to 0.95.
            temp (float, optional): The temperature value for sampling. Defaults to 1.25
            batch_size (int, optional): The batch size for generation. Defaults to 1.

        Returns:
            List[str]: A list of generated heavy chain sequences.

        """
        # generate backwards
        prompt = f"2{light_chain[::-1]}"
        return self.generate(
            num_return_sequences,
            top_p=top_p,
            temp=temp,
            batch_size=batch_size,
            backwards=True,
            prompt=prompt,
        )

    def generate_light_chain(
        self,
        heavy_chain: str,
        num_return_sequences: int,
        top_p=0.95,
        temp=1.25,
        batch_size=1,
    ) -> list[str]:
        """
        Given a heavy chain sequence, generates a light chain sequence.
        """
        # Generate forwards
        prompt = f"1{heavy_chain}"
        return self.generate(
            num_return_sequences,
            backwards=False,
            top_p=top_p,
            temp=temp,
            batch_size=batch_size,
            prompt=prompt,
        )

    def get_batch_log_likelihoods(
        self, sequences: list[str], batch_size: int = 32
    ) -> list[float]:
        """
        Computes the log likelihood for a batch of sequences.
        Ensures not caculated for padding tokens and the beggining and start
        tokens.

        Args:
            sequences (List[str]): A list of sequences
                for which to compute the likelihood.
            batch_size (int): The size of each batch for processing.
        Returns:
            likelihoods (List[float]): A list of log likelihoods for each sequence.
        """
        likelihoods = []

        # This required as not stored as bos and eos token in tokenizer
        bos_token_id = self.tokenizer.encode("1")[0]
        eos_token_id = self.tokenizer.encode("2")[0]
        pad_token_id = self.tokenizer.pad_token_id
        special_token_ids = [bos_token_id, eos_token_id, pad_token_id]

        # Split sequences into batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i : i + batch_size]

            # Tokenize all sequences once
            inputs = self.tokenizer(
                batch_sequences, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = inputs["input_ids"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits

            # Compute the likelihood for each sequence in the batch
            for input_id, logit in zip(input_ids, logits):
                # align the logits with the input ids
                # (remove first element of logits and last element of input_ids)
                # contiguous() makes sure stored contiguously in memory
                shift_logits = logit[:-1, :].contiguous()
                shift_labels = input_id[1:].contiguous().long()
                # make sure we don't do for special tokens, include 1, 2 and pad
                # (1 and 2 are start and end tokens)
                mask = torch.ones(shift_labels.shape, dtype=torch.bool).to(self.device)
                for token_id in special_token_ids:
                    mask = mask & (shift_labels != token_id)
                # Compute the negative log-likelihood using cross_entropy,
                # ignoring masked tokens
                nll = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1))[mask],
                    shift_labels.view(-1)[mask],
                    reduction="mean",
                )
                likelihoods.append(-nll)

        return torch.stack(likelihoods, dim=0).cpu().numpy()
