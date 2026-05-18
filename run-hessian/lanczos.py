'''
Acknowledgement: 
The code structure in this python file is referred to: https://github.com/vectozavr/llm-hessian/blob/main/src/single_layer_single_block.py
'''

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from filelock import FileLock
from torch.autograd.functional import vhp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from utils import (
    IGNORE_INDEX,
    check_gpus,
    disable_non_differential_modules,
    get_all_blocks,
    get_metamath_dataloader,
    get_nested_attr,
    group_add,
    group_product,
    move_to_device,
    normalization,
    set_seed,
)


def compute_eigen_for_layer(model, data_batches, layer_names_list, block_indices_list, max_iter, tol):
    target_layers = []
    target_params = []
    param_group_info = []

    for l_name, b_idx in zip(layer_names_list, block_indices_list):
        block = get_all_blocks(model)[b_idx]
        try:
            layer = get_nested_attr(block, l_name)
        except AttributeError:
            return None
        p = layer.weight.clone().requires_grad_(True)
        target_layers.append(layer)
        target_params.append(p)
        param_group_info.append(f"Block_{b_idx}.{l_name}")

    if not target_params:
        return None

    # Compute per-batch token counts for weighted HVP accumulation
    batch_token_counts = []
    for batch in data_batches:
        shift_labels = batch['labels'][..., 1:]
        batch_token_counts.append((shift_labels != IGNORE_INDEX).sum().item())

    total_valid_tokens = sum(batch_token_counts)
    batch_weights = [n / total_valid_tokens for n in batch_token_counts]
    print(f"Total valid tokens: {total_valid_tokens}, Batches: {len(data_batches)}")

    def compute_loss_on_batch(model, batch_data):
        device = target_params[0].device
        input_ids      = batch_data['input_ids'].to(device)
        labels         = batch_data['labels'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)

        logits       = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        n_valid = (shift_labels != IGNORE_INDEX).sum().item()
        loss = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=IGNORE_INDEX)(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss / n_valid

    def get_partial_ppl_fn(batch_idx):
        current_batch = data_batches[batch_idx]

        def partial_ppl_fn(*weights):
            original_forwards = []

            def make_custom_forward(weight_tensor):
                def custom_forward(self, inpt):
                    return F.linear(inpt, weight_tensor, self.bias)
                return custom_forward

            for i, layer in enumerate(target_layers):
                original_forwards.append(layer.forward)
                layer.forward = make_custom_forward(weights[i]).__get__(layer, type(layer))
            try:
                return compute_loss_on_batch(model, current_batch)
            finally:
                for i, layer in enumerate(target_layers):
                    layer.forward = original_forwards[i]

        return partial_ppl_fn

    # Initialise v0 on GPU; keep history on CPU to avoid VRAM exhaustion
    v_current_gpu = normalization([torch.randn_like(p) for p in target_params])
    v_list_cpu    = [move_to_device(v_current_gpu, 'cpu')]

    alphas, betas = [], []
    last_max_eig  = None
    converged_step = max_iter
    max_eigen_history, min_eigen_history = [], []
    stable_count = 0
    patience = 1

    desc = f"Layer {block_indices_list[0]} {layer_names_list[0].split('.')[1]}"
    pbar = tqdm(range(max_iter), desc=desc, unit="iter", leave=True)

    for i in pbar:
        # A. Accumulate HVP across all batches (GPU)
        Hv_accum_gpu = [torch.zeros_like(p) for p in target_params]
        for k in tqdm(range(len(data_batches)), desc=f" Iter {i+1} HVP", unit="batch", leave=False):
            _, vhp_result = vhp(get_partial_ppl_fn(k), tuple(target_params), tuple(v_current_gpu))
            Hv_accum_gpu = group_add(Hv_accum_gpu, vhp_result, alpha=batch_weights[k])

        # B. Offload to CPU for orthogonalization
        w_cpu    = move_to_device(Hv_accum_gpu, 'cpu')
        v_curr_c = v_list_cpu[-1]

        # C. Basic Lanczos step: alpha = v_i^T * w,  w = w - alpha*v_i
        alpha = group_product(v_curr_c, w_cpu)
        if torch.is_tensor(alpha): alpha = alpha.item()
        alphas.append(alpha)
        w_cpu = group_add(w_cpu, v_curr_c, alpha=-alpha)

        if i > 0:
            w_cpu = group_add(w_cpu, v_list_cpu[-2], alpha=-betas[-1])

        # D. Full re-orthogonalization (Gram-Schmidt against all past vectors)
        for v_past in v_list_cpu:
            overlap = group_product(w_cpu, v_past)
            w_cpu = group_add(w_cpu, v_past, alpha=-overlap)

        # E. Beta = ||w||
        beta_next = group_product(w_cpu, w_cpu)
        beta_next = beta_next.sqrt().item() if torch.is_tensor(beta_next) else np.sqrt(beta_next)

        if i < max_iter - 1:
            betas.append(beta_next)

        # F. Build tridiagonal matrix T and extract eigenvalues
        dim_t = len(alphas)
        T = torch.zeros(dim_t, dim_t)
        T.diagonal().copy_(torch.tensor(alphas))
        if betas:
            b = torch.tensor(betas[:dim_t - 1])
            T.diagonal(offset=1).copy_(b)
            T.diagonal(offset=-1).copy_(b)

        eigs = torch.linalg.eigvalsh(T)
        current_max = eigs.max().item()
        current_min = eigs.min().item()
        max_eigen_history.append(current_max)
        min_eigen_history.append(current_min)

        # G. Convergence check
        if last_max_eig is not None and i >= 4:
            if abs(current_max - last_max_eig) < tol:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= patience:
                converged_step = i + 1
                break

        last_max_eig = current_max
        pbar.set_postfix({"Max": f"{current_max:.3f}"})

        if beta_next < 1e-6:  # Lanczos breakdown
            converged_step = i + 1
            break

        # H. Prepare next Lanczos vector
        v_next_cpu = [wi / (beta_next + 1e-6) for wi in w_cpu]
        v_list_cpu.append(v_next_cpu)
        v_current_gpu = move_to_device(v_next_cpu, target_params[0].device)

    pbar.close()

    return {
        "param_group":        param_group_info,
        "max_eigen":          current_max,
        "min_eigen":          current_min,
        "converge_step":      converged_step,
        "total_valid_tokens": total_valid_tokens,
        "num_batches":        len(data_batches),
        "max_eigen_history":  max_eigen_history,
        "min_eigen_history":  min_eigen_history,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',          type=str,   required=True)
    parser.add_argument('--data_path',      type=str,   required=True)
    parser.add_argument('--dataset_field',  type=str,   default="query,response")
    parser.add_argument('--scan_modules',   type=str,   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument('--layer_start',    type=int,   default=0)
    parser.add_argument('--layer_end',      type=int,   default=-1)
    parser.add_argument('--max_iter',       type=int,   default=100)
    parser.add_argument('--tol',            type=float, default=1e-3)
    parser.add_argument('--b',              type=int,   default=10)
    parser.add_argument('--model_input_bs', type=int,   default=1)
    parser.add_argument('--seed',           type=int,   default=0)
    parser.add_argument('--cache_dir', type=str, default=None,
                            help='If None, use the default Hugging Face cache path.'
                        )
    parser.add_argument('--peft',           type=str,   default=None)
    parser.add_argument('--rank',           type=int,   default=8)

    args = parser.parse_args()
    check_gpus()
    set_seed(args.seed)
    disable_non_differential_modules()

    FULL_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, padding_side="right")
    tokenizer.model_max_length = 512
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.peft is not None:
        print(f"Applying {args.peft} config...")
        peft_lower = args.peft.lower()

        if peft_lower in ('lora', 'dora', 'pissa', 'olora'):
            model = AutoModelForCausalLM.from_pretrained(
                args.model, cache_dir=args.cache_dir,
                dtype=torch.float32, device_map="auto", trust_remote_code=True,
            )
            init_map = {'lora': True, 'dora': True, 'pissa': 'pissa_niter_16', 'olora': 'olora'}
            peft_config = LoraConfig(
                use_dora=(peft_lower == 'dora'),
                task_type=TaskType.CAUSAL_LM,
                target_modules=FULL_TARGET_MODULES,
                r=args.rank,
                lora_alpha=args.rank,  # lora_alpha == rank keeps effective scale at 1
                init_lora_weights=init_map[peft_lower],
            )
            model = get_peft_model(model, peft_config)

        elif peft_lower == 'loraga':
            paths = {
                "Qwen/Qwen3-0.6B-Base": f'../model_init/LoRAGA-qwen-3-0.6b-r{args.rank}-fp32',
            }
            res_path = paths.get(args.model)
            if res_path is None:
                raise NotImplementedError(f"LoRA-GA not configured for model: {args.model}")
            model = AutoModelForCausalLM.from_pretrained(
                res_path, dtype=torch.float32, device_map="auto", trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, res_path, subfolder="loraga_init", is_trainable=True)

        elif peft_lower == 'milora':
            paths = {
                "Qwen/Qwen3-0.6B-Base":     f'/storage/ssd3/ArthurLee/model_init/MiLoRA-qwen-3-0.6b-r{args.rank}-fp32',
                "google/gemma-3-1b-pt":     f'/storage/ssd3/ArthurLee/model_init/MiLoRA-gemma-3-1b-r{args.rank}-fp32',
                "meta-llama/Llama-2-7b-hf": f'/storage/ssd3/ArthurLee/model_init/MiLoRA-llama-2-7b-r{args.rank}-fp32',
            }
            res_path = paths.get(args.model)
            if res_path is None:
                raise NotImplementedError(f"MiLoRA not configured for model: {args.model}")
            model = AutoModelForCausalLM.from_pretrained(
                res_path, dtype=torch.float32, device_map="auto", trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, res_path, subfolder="milora_init", is_trainable=True)

        elif peft_lower == 'initab':
            paths = {
                "Qwen/Qwen3-0.6B-Base":     f'/storage/ssd3/ArthurLee/model_init/InitAB-qwen-3-0.6b-r{args.rank}-fp32',
                "google/gemma-3-1b-pt":     f'/storage/ssd3/ArthurLee/model_init/InitAB-gemma-3-1b-r{args.rank}-fp32',
                "meta-llama/Llama-2-7b-hf": f'/storage/ssd3/ArthurLee/model_init/InitAB-llama-2-7b-r{args.rank}-fp32',
            }
            res_path = paths.get(args.model)
            if res_path is None:
                raise NotImplementedError(f"InitAB not configured for model: {args.model}")
            model = AutoModelForCausalLM.from_pretrained(
                res_path, dtype=torch.float32, device_map="auto", trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, res_path, subfolder="init_initab", is_trainable=True)

    print(model)
    model.eval()

    q_field, r_field = [x.strip() for x in args.dataset_field.split(',')]
    data_batches = list(get_metamath_dataloader(
        tokenizer, args.data_path,
        num_samples=args.b, batch_size=args.model_input_bs,
        seed=args.seed, query_field=q_field, response_field=r_field,
    ))

    num_layers      = model.config.num_hidden_layers
    l_start         = args.layer_start
    l_end           = args.layer_end if args.layer_end != -1 else num_layers
    modules_to_scan = [x.strip() for x in args.scan_modules.split(',')]

    os.makedirs("./eigen_results", exist_ok=True)
    from pathlib import Path
    dataset_name = Path(args.data_path).parent.name
    json_path = os.path.join(
        "./eigen_results",
        f"{args.model.split('/')[-1]}-{dataset_name}"
        f"-{args.peft}-r{args.rank}-b{args.b}-seed{args.seed}-bs{args.model_input_bs}.json",
    )
    lock_path = json_path + ".lock"
    print(f'Hessian results will be saved to {json_path}')

    processed_keys = set()
    with FileLock(lock_path, timeout=10):
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                for r in json.load(f):
                    processed_keys.add(r["param_group"][0].split(".lora_B")[0])

    print(f"Scanning layers {l_start} to {l_end - 1}, modules: {modules_to_scan}")
    for layer_idx in range(l_start, l_end):
        for mod_name in modules_to_scan:
            prefix      = "self_attn" if mod_name in ["q_proj", "k_proj", "v_proj", "o_proj"] else "mlp"
            current_key = f"Block_{layer_idx}.{prefix}.{mod_name}"

            if current_key in processed_keys:
                print(f"Skipping {current_key} (already exists)")
                continue

            report = compute_eigen_for_layer(
                model, data_batches,
                [f"{prefix}.{mod_name}.lora_B.default", f"{prefix}.{mod_name}.lora_A.default"],
                [layer_idx, layer_idx],
                args.max_iter, args.tol,
            )

            if report:
                with FileLock(lock_path, timeout=10):
                    fresh_results = []
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            fresh_results = json.load(f)
                    fresh_keys = {r["param_group"][0].split(".lora_B")[0] for r in fresh_results}
                    if current_key not in fresh_keys:
                        fresh_results.append(report)
                        with open(json_path, 'w') as f:
                            json.dump(fresh_results, f, indent=4)
                processed_keys.add(current_key)

    print(f"Done. Results saved to {json_path}")