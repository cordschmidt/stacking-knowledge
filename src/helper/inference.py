"""
Utility functions for running model inference, specifically for computing
perplexity using causal language models
"""

from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING, Dict, List

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    # avoid circular imports
    from src.custom_trainer import CustomTrainer

def prepare_dataset_for_ppl_inference(
    trainer: CustomTrainer,
    dataset: Dataset,
) -> Dataset:
    """
    Preprocess dataset to remove columns that are not used by the perplexity computation's
    forward pass through the trainer model.

    Args:
        trainer: a CustomTrainer object used for training the model
        dataset: the dataset that will be scored for perplexity
    """
    # Get info from trainer which columns can be ignored for inference
    ignore_columns = trainer._get_ignore_columns(dataset)

    # Return dataset without unused columns (but keeping special_tokens_mask)
    return dataset.remove_columns(ignore_columns)

def compute_trainer_perplexity(
    batch: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizerFast,
    trainer: CustomTrainer,
) -> List[float]:
    """
    Computes total negative log-likelihood and valid token count for a batch,
    following the Hugging Face recommendation for accurate PPL aggregation (see https://huggingface.co/docs/transformers/perplexity)
    """
    input_ids = batch["input_ids"].to(trainer.args.device)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(trainer.args.device)

    # Mask padding tokens with -100 so they are ignored by the loss function.
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        outputs = trainer.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # loss is the average NLL per non-masked token in the batch
        neg_log_likelihood = outputs.loss

    # Count valid tokens (those not masked by -100)
    num_valid_tokens = (labels != -100).sum().item()

    # Total NLL for this batch = (Average NLL) * (Number of valid tokens)
    total_batch_nll = neg_log_likelihood.item() * num_valid_tokens

    return total_batch_nll, num_valid_tokens
