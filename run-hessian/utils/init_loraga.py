"""
LoRA-GA Initialization Script
Following the PiSSA/MiLoRA pattern: save residual model + init adapter separately.

This script:
1. Loads the base model
2. Loads a small batch of training data for gradient estimation
3. Estimates gradients using LoRA-GA's method
4. Applies LoRA-GA initialization (modifies base weights + creates adapter)
5. Saves the init adapter to output_dir/loraga_init/
6. Saves the residual model (modified base) to output_dir/

Usage:
    python utils/init_loraga.py \
        --device cuda:0 \
        --base_model_path meta-llama/Llama-2-7b-hf \
        --output_dir output/LoRA-GA-Llama-2-7b-r128 \
        --data_path pissa-dataset \
        --sub_task metamath \
        --lora_r 128 \
        --lora_alpha 128 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

Note:
LoRA-GA has handled the potential precision issue itslef here:
https://github.com/Outsider565/LoRA-GA/blob/c4cd5372c75b290924214b348008891f744512ef/peft/src/peft/tuners/lora/layer.py#L171-L299

"""

import torch
import os
import re
import json
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraGAConfig, get_peft_model
from peft.utils.lora_ga_utils import estimate_gradient, LoraGAContext
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_seed(42)

IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


class GradientEstimationDataset(Dataset):
    """
    Simple dataset for gradient estimation.
    Tokenizes instruction/output pairs from the training data.
    """
    def __init__(self, raw_dataset, tokenizer, max_length=512, num_samples=32):
        self.samples = []
        n = min(num_samples, len(raw_dataset))
        for idx in range(n):
            example = raw_dataset[idx]
            source = PROMPT.format(instruction=example["instruction"])
            target = example["output"] + tokenizer.eos_token
            text = source + target

            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            # Mask instruction part in labels
            source_enc = tokenizer(source, max_length=max_length, truncation=True)
            source_len = len(source_enc["input_ids"])
            labels = input_ids.clone()
            labels[:source_len] = IGNORE_INDEX
            labels[attention_mask == 0] = IGNORE_INDEX

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def load_raw_dataset(data_path, sub_tasks, max_samples=None):
    """Load raw dataset for gradient estimation (same logic as train.py)."""
    all_data = []
    for task in sub_tasks:
        # Clean task name: remove ":num_samples" and "-epN" suffixes
        task_clean = task.split(":")[0]
        task_clean = re.sub(r"-ep\d+$", "", task_clean)

        if task_clean == "commonsense":
            cs_path = os.path.join(data_path, "commonsense_15k.json")
            if not os.path.exists(cs_path):
                raise FileNotFoundError(f"Cannot find commonsense data at {cs_path}")
            with open(cs_path, "r", encoding="utf-8") as f:
                cs_data = json.load(f)
            from datasets import Dataset as HFDataset
            ds = HFDataset.from_list(cs_data)
            all_data.append(ds)
        else:
            ds = load_dataset(data_path, data_dir=task_clean, split="train")
            all_data.append(ds)

    dataset = concatenate_datasets(all_data)
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="LoRA-GA initialization: estimate gradients, init adapters, save residual model"
    )
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # Data arguments (needed for gradient estimation)
    parser.add_argument("--data_path", type=str, default="pissa-dataset")
    parser.add_argument("--sub_task", nargs="+", required=True)
    # Model precision
    parser.add_argument("--bits", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    # LoRA-GA arguments
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--target_modules", nargs="+", required=True)
    parser.add_argument("--stable_gamma", type=int, default=16,
                        help="LoRA-GA stable scaling gamma (paper default: 16 for small models, 64 for 7B+)")
    # Gradient estimation arguments
    parser.add_argument("--grad_num_samples", type=int, default=32,
                        help="Number of samples for gradient estimation")
    parser.add_argument("--grad_batch_size", type=int, default=2,
                        help="Micro-batch size for gradient estimation")
    parser.add_argument("--model_max_length", type=int, default=512)
    args = parser.parse_args()
    print(args)

    # =========================================================
    # 1. Load base model
    # =========================================================
    logger.info(f"Loading base model from {args.base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=(
            torch.float16 if args.bits == "fp16"
            else (torch.bfloat16 if args.bits == "bf16" else torch.float32)
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = args.model_max_length

    # =========================================================
    # 2. Prepare data for gradient estimation
    # =========================================================
    logger.info(f"Loading data for gradient estimation (num_samples={args.grad_num_samples})...")
    raw_dataset = load_raw_dataset(
        args.data_path, args.sub_task, max_samples=args.grad_num_samples
    )
    grad_dataset = GradientEstimationDataset(
        raw_dataset, tokenizer,
        max_length=args.model_max_length,
        num_samples=args.grad_num_samples,
    )
    dataloader = DataLoader(grad_dataset, batch_size=args.grad_batch_size)
    logger.info(f"Gradient estimation dataset: {len(grad_dataset)} samples, "
                f"batch_size={args.grad_batch_size}, "
                f"num_batches={len(dataloader)}")

    # =========================================================
    # 3. Setup accelerator & move model to device
    # =========================================================
    accelerator = Accelerator()
    model = model.to(accelerator.device)
    dataloader = accelerator.prepare(dataloader)

    # =========================================================
    # 4. LoRA-GA config
    # =========================================================
    logger.info(f"LoRA-GA config: r={args.lora_r}, alpha={args.lora_alpha}, "
                f"gamma={args.stable_gamma}, targets={args.target_modules}")
    peft_config = LoraGAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        stable_gamma=args.stable_gamma,
    )

    # =========================================================
    # 5. Estimate gradients
    # =========================================================
    logger.info("Estimating gradients...")
    named_grad = estimate_gradient(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
        quant_flag=False,
    )
    logger.info(f"Gradient estimation done. Total gradient keys: {len(named_grad)}")

    # =========================================================
    # 6. Apply LoRA-GA initialization
    #    This does two things:
    #    - Initializes lora_A, lora_B from gradient SVD
    #    - Modifies base weights: W_init = W_0 - η*B*A (residual)
    # =========================================================
    logger.info("Applying LoRA-GA initialization (gradient approximation + stable scale)...")
    with LoraGAContext(model=model, named_grad=named_grad):
        model = get_peft_model(model=model, peft_config=peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    # =========================================================
    # 7. Save init adapter (like PiSSA saves to pissa_init/)
    #    Set init_lora_weights=True so that PeftModel.from_pretrained
    #    loads the saved weights without re-doing LoRA-GA init.
    # =========================================================
    init_adapter_dir = os.path.join(args.output_dir, "loraga_init")
    logger.info(f"Saving LoRA-GA init adapter to {init_adapter_dir}...")
    model.peft_config["default"].init_lora_weights = True
    model.save_pretrained(init_adapter_dir, safe_serialization=False)

    # =========================================================
    # 8. Save residual model (base with modified weights)
    #    unload() strips the LoRA structure, leaving base weights
    #    as the residual (W_init = W_0 - η*B*A)
    # =========================================================
    logger.info(f"Saving residual model to {args.output_dir}...")
    base_model = model.unload()
    base_model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("=" * 60)
    logger.info("LoRA-GA initialization complete!")
    logger.info(f"  Residual model : {args.output_dir}")
    logger.info(f"  Init adapter   : {init_adapter_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()