import string
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data.sampler import Sampler
from transformers import PreTrainedTokenizerFast

from src.config import BabyLMConfig


def base_collate_fn(_samples: List[Dict[str, List[Tuple[int, float]]]]):
    """
    Combines a list of samples into a batch of tensors.
    """
    joined_batch = defaultdict(list)
    for sample in _samples:
        for key, val in sample.items():
            joined_batch[key].append(torch.tensor(val))

    batch = {}

    for key, val in joined_batch.items():
        batch[key] = torch.stack(val)

    return batch


class SequentialSubsetSampler(Sampler):
    """
    Samples elements sequentially from a set of indices, always in the same order.
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DatasetPreprocessor:
    """
    Preprocessing class that tokenizes input text and chunks it into fixed-length inputs.
    """

    def __init__(self, cfg: BabyLMConfig, tokenizer: PreTrainedTokenizerFast):
        """
        Args:
            cfg: Configuration object with preprocessing options.
            tokenizer: Hugging Face tokenizer instance.
        """
        self.include_punctuation = cfg.data_preprocessing.include_punctuation
        self.max_input_length = cfg.data_preprocessing.max_input_length
        self.join_sentences = cfg.data_preprocessing.join_sentences
        self.callback_functions = cfg.data_preprocessing.callback_functions
        self.dataset_subconfig = cfg.dataset.subconfig
        self.tokenizer = tokenizer

    def __call__(self, examples):
        """
        Process a batch of text examples:
            - Optionally remove punctuation
            - Tokenize text
            - Join or split sentences based on config
            - Return input_ids, attention_mask, etc.
        """

        if not self.include_punctuation:
            examples["text"] = [
                line.translate(str.maketrans("", "", string.punctuation))
                for line in examples["text"]
            ]

        batch = {
            "input_ids": [],
            "special_tokens_mask": [],
            "attention_mask": [],
            "filename": [],
        }

        full_tokenized_inputs = {
            "input_ids": [],
            "special_tokens_mask": [],
            "attention_mask": [],
            "filename": [],
        }

        for example in range(len(examples["text"])):
            text = examples["text"][example]
            filename = examples["filename"][example]

            # TODO: Adjust this to LLaMA models? How does padding have to be changed here?
            tokenized_inputs = self.tokenizer(
                text,
                pad_to_multiple_of=self.max_input_length if not self.join_sentences else None,
                padding="longest" if not self.join_sentences else "do_not_pad",
                max_length=self.max_input_length if not self.join_sentences else None,
                truncation=False,
                return_special_tokens_mask=True,
            )

            if self.join_sentences:
                # If we're joining all sentences into one long sequence before chunking,
                # extend each token field (input_ids, attention_mask, etc.) into a growing list
                for field_name in ["input_ids", "special_tokens_mask", "attention_mask"]:
                    full_tokenized_inputs[field_name].extend(tokenized_inputs[field_name])

                # Store the filename repeatedly for each token, to keep alignment
                full_tokenized_inputs["filename"].extend(
                    [filename] * len(tokenized_inputs["input_ids"])
                )
            else:
                # If we're not joining, split each tokenized sentence into fixed-length chunks immediately
                for i in range(0, len(tokenized_inputs["input_ids"]), self.max_input_length):
                    # Skip any chunk that contains only special tokens (e.g. [PAD], [CLS])
                    if sum(tokenized_inputs["special_tokens_mask"][
                           i:i + self.max_input_length]) == self.max_input_length:
                        break
                    # Add a chunk of each token field to the batch
                    for field_name in ["input_ids", "special_tokens_mask", "attention_mask"]:
                        batch[field_name].append(
                            tokenized_inputs[field_name][i:i + self.max_input_length]
                        )
                    # Store the filename once per chunk
                    batch["filename"].append(filename)


        if self.join_sentences:
            # Compute the maximum length we can divide evenly into chunks of `max_input_length`
            truncated_length = (
                len(full_tokenized_inputs["input_ids"]) // self.max_input_length
            ) * self.max_input_length

            # Iterate over the long tokenized input in fixed-size steps
            for i in range(0, truncated_length, self.max_input_length):
                # For each field, extract a chunk and add it to the batch
                for field_name in ["input_ids", "special_tokens_mask", "attention_mask"]:
                    batch[field_name].append(full_tokenized_inputs[field_name][i:i + self.max_input_length])
                batch["filename"].append(full_tokenized_inputs["filename"][i])

        if self.callback_functions:
            for callback_function in self.callback_functions:
                examples[callback_function] = getattr(self, callback_function)(examples)

        return batch
