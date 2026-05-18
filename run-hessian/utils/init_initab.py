import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import argparse
import logging
import torch
import transformers
from datetime import datetime
from peft import LoraConfig, TaskType, get_peft_model
from utils.init_AB_util import modify_initAB_model

# Setting up Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    
    final_output_dir = f"{args.output_dir}_{timestamp}"    
    adapter_save_dir = os.path.join(final_output_dir, "init_initab")
    
    residual_save_dir = final_output_dir
    os.makedirs(adapter_save_dir, exist_ok=True)

    print('='*80)
    print(f'Starting InitAB Initialization Process')
    print(f'Timestamp: {timestamp}')
    print(f'Base Model: {args.base_model_path}')
    print(f'Final Output Root (Residual): {residual_save_dir}')
    print(f'Adapter Subfolder: {adapter_save_dir}')
    print('='*80)

    # ================= [Load Model] =================
    print('Loading pretrained huggingface model...')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model_path, 
        torch_dtype=(
            torch.float16
            if args.bits == "fp16"
            else (torch.bfloat16 if args.bits == "bf16" else torch.float32)
        ),
        device_map=args.device,
    )

    # [REQ 2] Print structure
    print("\n" + "-"*30)
    print("Stage 1: Base Model Structure")
    print("-"*30)
    print(model)
    print("-"*30 + "\n")

    ref_weight = None
    ref_name = None
    target_list = args.target_modules if isinstance(args.target_modules, list) else [args.target_modules]
    target_module_key = "q_proj" if "q_proj" in target_list else target_list[0]
    
    for name, module in model.named_modules():
        if target_module_key in name and isinstance(module, torch.nn.Linear):
            ref_weight = module.weight.detach().clone().cpu() # 備份原始權重到 CPU
            ref_name = name
            
            # [REQ 1] Print first 5 parameters
            top5_params = ref_weight.view(-1)[:5].tolist()
            print(f"[Check] Reference weight captured from: {name}")
            print(f"[Check] Top-5 Params: {top5_params}")
            break
    
    if ref_weight is None:
        logger.warning("Could not find a target module to set as reference. Verification will be skipped.")
    # ========================================================

    # ================= [Configure PEFT] =================
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.target_modules, 
        r=args.lora_r, 
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout    
    )
    
    model = get_peft_model(model, peft_config)
    print(f'PEFT Model created. Trainable parameters:')
    model.print_trainable_parameters()

    # [REQ 2] Print structure
    print("\n" + "-"*30)
    print("Stage 2: PEFT Model Structure (with LoRA)")
    print("-"*30)
    print(model)
    print("-"*30 + "\n")

    # ================= [Modify InitAB] =================
    # This modifies the base model weights in-place to W_res = W - AB
    print(f'Modifying weights for correct InitAB (Non-deterministic)...')
    modify_initAB_model(model, "AB_1_1_RESET")

    # ================= [Save Adapter] =================
    # Save the A and B matrices to the subfolder
    logger.info(f"Saving InitAB Adapter to {adapter_save_dir} ...")
    model.save_pretrained(adapter_save_dir, safe_serialization=False)
    print(f'Adapter saved.')

    # ================= [Prepare Residual Model] =================
    print(f'Preparing to save InitAB residual base model...')
    print("[Action] Unloading adapter from model to get residual base...")
    res_model = model.unload() 


    # [REQ 2] Print structure
    print("\n" + "-"*30)
    print("Stage 3: Unloaded Model Structure (Should be Residual)")
    print("-"*30)
    print(res_model)
    print("-"*30 + "\n")
    
    # ================= [Check Logic Part 2] =================
    print("-" * 60)
    print("VERIFYING RESIDUAL MODEL WEIGHTS...")
    
    save_safe_flag = False 
    
    if ref_weight is not None:
        try:
            new_module = res_model
            for atom in ref_name.split('.'):
                new_module = getattr(new_module, atom)
            
            new_weight = new_module.weight.detach().cpu()
            
            ref_top5 = ref_weight.view(-1)[:5].tolist()
            new_top5 = new_weight.view(-1)[:5].tolist()

            diff = (ref_weight - new_weight).abs().sum().item()
            
            print(f" > Original Top-5 Params: {ref_top5}")
            print(f" > Unloaded Top-5 Params: {new_top5}")
            print(f" > Total L1 Difference:   {diff:.6f}")
            
            if diff < 1e-4:
                print("\n[ALERT] Difference is nearly ZERO!")
                print(" -> This means .unload() MERGED the LoRA weights back to original.")
                print(" -> The model is effectively the ORIGINAL PRETRAINED MODEL.")
                save_safe_flag = False
            else:
                print("\n[SUCCESS] Significant difference detected.")
                print(" -> The LoRA weights were NOT merged.")
                print(" -> We successfully obtained the RESIDUAL model.")
                save_safe_flag = True
                
        except Exception as e:
            print(f"[Check Error] Could not verify weights: {e}")
            save_safe_flag = False
    
    print("-" * 60)
    # ========================================================

    # ================= [Save Residual Model] =================
    print(f'Saving InitAB residual to {residual_save_dir}')
    
    if save_safe_flag:
        print("Using the unloaded model (Verification Passed).")
        res_model.save_pretrained(residual_save_dir, safe_serialization=False)
    else:
        raise ValueError("Residual verification failed! Aborting save to prevent overwriting with incorrect model.")

    # ================= [Save Tokenizer] =================
    print(f'Saving Tokenizer to {residual_save_dir}')
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(residual_save_dir)
    
    logger.info(f'Finished! All artifacts saved to {final_output_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for InitAB Initialization and Saving")    
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the original pretrained base model.")
    parser.add_argument("--device", type=str, default="auto", help="Device map, e.g., 'auto', 'cuda:0'.")
    parser.add_argument("--bits", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Precision for loading the model.")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA Rank.")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA Alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0, help="LoRA Dropout.")
    parser.add_argument('--target_modules', nargs='+', help='List of target modules (e.g., q_proj v_proj)', required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save results. A timestamp will be appended.")

    args = parser.parse_args()
    
    main(args)