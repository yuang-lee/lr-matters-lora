import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed
from peft import LoraConfig, TaskType, get_peft_model
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_seed(42)

total_time=0

def initialize_lora_layer(weights, rank, init_weights="milora"):
    start = time.time()
    U, S, V = torch.linalg.svd(weights, full_matrices=False)
    end = time.time()
    delta_time = end - start
    logger.info(f"delta_time for linglg.svd{delta_time}")
    global total_time
    total_time += end - start
    lora_alpha=rank
    if init_weights == "milora":
        U_select = U[:, -rank:]
        S_select = S[-rank:]
        V_select = V[-rank:, :]
    # if mode == "min":
    #     U_select = U[:, -rank:]
    #     S_select = S[-rank:]
    #     V_select = V[-rank:, :]
    # elif mode == "mid":
    #     mid_start = (len(S) - rank) // 2
    #     mid_end = mid_start + rank
    #     U_select = U[:, mid_start:mid_end]
    #     S_select = S[mid_start:mid_end]
    #     V_select = V[mid_start:mid_end, :]
    # elif mode == "max":
    #     U_select = U[:, :rank]
    #     S_select = S[:rank]
    #     V_select = V[:rank, :]
    # elif mode == "random":
    #     indices = np.random.choice(len(S), rank, replace=False)
    #     indices = np.sort(indices)
    #     U_select = U[:, indices]
    #     S_select = S[indices]
    #     V_select = V[indices, :]
    else:
        raise ValueError("Unknown mode!")

    scaling = lora_alpha / rank
    S_select /= scaling  
    S_sqrt = torch.sqrt(S_select)
    B = U_select @ torch.diag(S_sqrt)
    A = torch.diag(S_sqrt) @ V_select
    delta = scaling * B @ A

    return A, B, delta


def move_lora_file(SAVE_PATH):
    import os
    import shutil
    target_path = SAVE_PATH
    lora_path = os.path.join(target_path, 'milora_init')
    os.makedirs(lora_path, exist_ok=True)

    files_to_move = ['adapter_config.json', 'adapter_model.bin']
    for file_name in files_to_move:
        src_file = os.path.join(target_path, file_name)
        dst_file = os.path.join(lora_path, file_name)
        if os.path.exists(src_file):
            shutil.move(src_file, dst_file)
            print(f"Moved {src_file} to {dst_file}")
        else:
            print(f"{src_file} does not exist")

    print("Files moved successfully.")
    return

def svd_tailor_and_save(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, 
        dtype=(
            torch.float16
            if args.bits == "fp16"
            else (torch.bfloat16 if args.bits == "bf16" else torch.float32)
        ),
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,    
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config).to(args.device)

    logger.info(f"Start processing SVD and Lora initialization...")
    import time
    start = time.time()
    last_time = start
    with torch.no_grad():
        for n, p in model.named_parameters():
            if any(proj in n for proj in args.target_modules) and "lora" not in n:
                parent_name = n.split(".base_layer.weight")[0]
                parent_module = model.get_submodule(parent_name)
                lora_A_init, lora_B_init, delta = initialize_lora_layer(p.data.float(), args.lora_r, init_weights=args.init_weights)
                
                parent_module.base_layer.weight.data -= delta
                parent_module.lora_A['default'].weight.data = lora_A_init
                parent_module.lora_B['default'].weight.data = lora_B_init
                current = time.time()
                logger.info(f"processed: {parent_name} init_weights:{args.init_weights} svd_rank:{args.lora_r} time cost:{current-last_time}")
                last_time = current

    end = time.time()
    logger.info(f"Total time cost for SVD: {end-start}")
    global total_time
    logger.info(f"Specific time cost for SVD: {total_time}")

    logger.info(f"Save SVD model and initial lora to {args.output_dir}...")
    model.save_pretrained(args.output_dir+"/milora_init", safe_serialization=False)
    base_model = model.unload()
    base_model.save_pretrained(args.output_dir, safe_serialization=False)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info(f"Finished!")



import argparse

parser = argparse.ArgumentParser(description="argparse for parallel svd tailor")
parser.add_argument("--device", type=str, required=True, help="The device to host the model.")
parser.add_argument("--base_model_path", type=str, required=True, help="The name or path of the base model.")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--bits", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
parser.add_argument("--init_weights", type=str, default="milora")
parser.add_argument("--lora_r", type=int, default=128)
parser.add_argument("--lora_alpha", type=int, default=128)
parser.add_argument("--lora_dropout", type=float, default=0)
parser.add_argument('--target_modules', nargs='+', help='', required=True)
args = parser.parse_args()

svd_tailor_and_save(args=args)