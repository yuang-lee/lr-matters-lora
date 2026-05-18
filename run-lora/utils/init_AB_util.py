import torch.nn as nn
import torch
import os 
from peft import PeftModel
from peft.tuners.lora import Linear as LoraLinear
from tqdm import tqdm

def modify_initAB_model(model, init_para):
   lora_count = 0
   for module in model.modules():
       if isinstance(module, LoraLinear):
           if hasattr(module, 'lora_A'):
            #    print(module.lora_A)
               
               if "AB" in init_para:
                   mag_A = float(init_para.split("_")[1])
                   mag_B = float(init_para.split("_")[2])
                   fan_in = module.in_features
                   bound_A = mag_A / (fan_in ** 0.5)
                   bound_B = mag_B / (fan_in ** 0.5)
                #    print(f"bound_A: {bound_A}, bound_B: {bound_B}")
                   nn.init.uniform_(module.lora_A.default.weight, a=-bound_A, b=bound_A)
                   nn.init.uniform_(module.lora_B.default.weight, a=-bound_B, b=bound_B)

               if "RESET" in init_para:
                #    print("add -scaling*AB to W")
                   previous_dtype = module.weight.dtype
                   matmul_output = module.lora_B.default.weight.data @ module.lora_A.default.weight.data
                   matmul_output = matmul_output.T if module.fan_in_fan_out else matmul_output
                #    print(f"scaling: {module.scaling['default']}")
                   module.weight.data -= matmul_output.to(previous_dtype) * module.scaling['default']
               
               lora_count += 1
   
   print(f"Total modified LoRA modules: {lora_count}")