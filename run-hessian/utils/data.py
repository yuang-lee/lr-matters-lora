import copy
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader


IGNORE_INDEX = -100

# Instruction-following prompt template (Alpaca-style)
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


# =============================================================================
# Tokenisation
# =============================================================================

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized = [
        tokenizer(text, max_length=tokenizer.model_max_length, truncation=True)
        for text in strings
    ]
    input_ids = [np.array(t.input_ids) for t in tokenized]
    lengths = [len(t.input_ids) for t in tokenized]
    return dict(input_ids=input_ids, labels=input_ids, input_ids_lens=lengths, labels_lens=lengths)


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Tokenise source+target pairs and mask source tokens in labels."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tok, sources_tok = (
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    )
    input_ids = examples_tok["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tok["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def train_tokenize_function(examples, tokenizer, query_field: str, response_field: str) -> Dict:
    sources = [PROMPT.format_map(dict(instruction=q)) for q in examples[query_field]]
    targets = [f"{r}\n{tokenizer.eos_token}" for r in examples[response_field]]
    return preprocess(sources, targets, tokenizer)


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(inst["input_ids"]) for inst in instances]
        labels    = [torch.tensor(inst["labels"])    for inst in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# =============================================================================
# DataLoader
# =============================================================================

def get_metamath_dataloader(
    tokenizer,
    data_path: str,
    num_samples: int,
    batch_size: int,
    seed: int,
    query_field: str = "query",
    response_field: str = "response",
) -> DataLoader:
    """Load a supervised dataset and return a DataLoader ready for Hessian estimation."""
    print(f"Loading dataset from {data_path}...")
    if data_path.endswith('.json'):
        raw_dataset = load_dataset("json", data_files={"train": data_path}, split="train")
    else:
        raw_dataset = load_dataset(data_path, split="train")

    num_samples = min(num_samples, len(raw_dataset))
    raw_dataset = raw_dataset.shuffle(seed=seed).select(range(num_samples))

    tokenized = raw_dataset.map(
        lambda x: train_tokenize_function(x, tokenizer, query_field, response_field),
        batched=True,
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
        desc="Tokenizing",
    )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return DataLoader(tokenized, shuffle=False, batch_size=batch_size, collate_fn=collator)