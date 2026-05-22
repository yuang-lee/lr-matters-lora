# LoRA Hessian Analysis

## Environment Setup

We use the same environment as in <a href="../run-lora"><code>./run-lora</code></a>:

```bash
conda create -n lora-env python=3.12 -y
conda activate lora-env

# Install PyTorch (adjust for your CUDA version — see https://pytorch.org/get-started/previous-versions/)
pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

pip3 install -r requirements.txt
```

## Run Lanczos Algorithm

> [!NOTE]
> Due to the numerical instability of the Lanczos algorithm under finite-precision arithmetic, both the base model and LoRA adapters must use **Float32** precision.

### Vanilla LoRA

Run the Lanczos algorithm to estimate the maximum Hessian eigenvalues for Vanilla LoRA:

```bash
cd run-hessian
bash scripts/run.sh \
    --model Qwen/Qwen3-0.6B-Base \
    --peft lora \
    --data_path ../run-lora/lora-dataset/pissa-dataset/metamath/train.json \
    --dataset_field instruction,output \
    --rank 128 \
    --b 500 \
    --model_input_bs 5 \
    --max_iter 300 
```

All available options:

| Option | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B-Base` | HuggingFace model ID or local path |
| `--peft` | `lora` | Specific LoRA methods to estimate Hessian |
| `--data_path` | `./pissa-dataset/metamath/train.json` | Path to dataset JSON file |
| `--dataset_field` | `instruction,output` | Comma-separated query and response field names |
| `--scan_modules` | `q_proj,...,down_proj` | Comma-separated list of modules to scan |
| `--layer_start` | `0` | First transformer layer to scan (inclusive) |
| `--layer_end` | `-1` | Last transformer layer to scan (exclusive); `-1` scans all layers |
| `--b` | `500` | Total number of data samples |
| `--model_input_bs` | `5` | Batch size for Hessian calculation |
| `--max_iter` | `300` | Maximum Lanczos iterations |
| `--rank` | `128` | LoRA rank |
| `--tol` | `5e-3` | Convergence tolerance on the largest eigenvalue |
| `--cuda_devices` | `0,1` | `CUDA_VISIBLE_DEVICES` setting |

Results are saved as JSON files under `./eigen_results/`. An example of our results:
```
[
    {
        "param_group": [
            "Block_0.self_attn.q_proj.lora_B.default",
            "Block_0.self_attn.q_proj.lora_A.default"
        ],
        "max_eigen": 0.3809458911418915,
        "min_eigen": -0.2882091701030731,
        "converge_step": 9,
        "total_valid_tokens": 87284,
        "num_batches": 100,
        "max_eigen_history": [
            -7.856079946577665e-07,
            0.004104231018573046,
            0.06284769624471664,
            0.08257415890693665,
            0.13210350275039673,
            0.3137061595916748,
            0.3733288645744324,
            0.3803611695766449,
            0.3809458911418915 --> This is the final esitimated maximum Hessian eigenvalue
        ],
        "min_eigen_history": [
            -7.856079946577665e-07,
            -0.006510655861347914,
            -0.09462464600801468,
            -0.22672350704669952,
            -0.27700427174568176,
            -0.2861711382865906,
            -0.2878881096839905,
            -0.2881666421890259,
            -0.2882091701030731
        ]
    }
]
```

### LoRA Initialization Variants

#### *OLoRA*

```bash
bash scripts/run.sh \
    --model Qwen/Qwen3-0.6B-Base \
    --peft olora \
    --data_path ../run-lora/lora-dataset/pissa-dataset/metamath/train.json \
    --dataset_field instruction,output \
    --rank 128 \
    --b 500 \
    --model_input_bs 5 \
    --max_iter 300 
```

#### *PiSSA*

```bash
bash scripts/run.sh \
    --model Qwen/Qwen3-0.6B-Base \
    --peft pissa \
    --data_path ../run-lora/lora-dataset/pissa-dataset/metamath/train.json \
    --dataset_field instruction,output \
    --rank 128 \
    --b 500 \
    --model_input_bs 5 \
    --max_iter 300 
```


> [!NOTE]
> For MiLoRA, Init[AB], and LoRA-GA, the model weights (residual weights and LoRA adapters) need to be pre-computed and saved locally before running the Hessian analysis. The expected paths are hard-coded in both `utils/init_XXX` and `lanczos.py` — adjust them to your custom path.


#### *MiLoRA*

```bash
python3 utils/init_milora.py \
      --device "cuda:0" \
      --base_model_path "Qwen/Qwen3-0.6B-Base" \
      --output_dir "./model_init/MiLoRA-qwen-3-0.6b-r128-fp32" \
      --init_weights "milora" \
      --lora_r "128" \
      --lora_alpha "128" \
      --lora_dropout 0 \
      --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
      --bits fp32 


bash scripts/run.sh \
    --model Qwen/Qwen3-0.6B-Base \
    --peft milora \
    --data_path ../run-lora/lora-dataset/pissa-dataset/metamath/train.json \
    --dataset_field instruction,output \
    --rank 128 \
    --b 500 \
    --model_input_bs 5 \
    --max_iter 300 
```



#### *Init[AB]*

```bash
python3 utils/init_initab.py \
      --device "cuda:0" \
      --base_model_path "Qwen/Qwen3-0.6B-Base" \
      --output_dir "./model_init/InitAB-qwen-3-0.6b-r128-fp32" \
      --lora_r "128" \
      --lora_alpha "128" \
      --lora_dropout 0 \
      --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
      --bits fp32 


bash scripts/run.sh \
    --model Qwen/Qwen3-0.6B-Base \
    --peft initab \
    --data_path ../run-lora/lora-dataset/pissa-dataset/metamath/train.json \
    --dataset_field instruction,output \
    --rank 128 \
    --b 500 \
    --model_input_bs 5 \
    --max_iter 300 
```

#### *LoRA-GA*

```bash
python3 utils/init_loraga.py \
      --base_model_path "Qwen/Qwen3-0.6B-Base" \
      --device "cuda:0" \
      --output_dir "./model_init/LoRAGA-qwen-3-0.6b-r128-fp32" \
      --data_path /storage/ssd1/ArthurLee/robust_PEFT/LoRA-related/weights_init/pissa-dataset \
      --sub_task metamath \
      --lora_r 128 \
      --lora_alpha 128 \
      --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
      --bits fp32 


bash scripts/run.sh \
    --model Qwen/Qwen3-0.6B-Base \
    --peft milora \
    --data_path ../run-lora/lora-dataset/pissa-dataset/metamath/train.json \
    --dataset_field instruction,output \
    --rank 128 \
    --b 500 \
    --model_input_bs 5 \
    --max_iter 300 
```