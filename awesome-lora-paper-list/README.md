# Awesome LoRA Paper List

<p align="center">
  <img src="./lora-paper-counts.png" alt="LoRA Paper Counts" width="520"/>
</p>

> [!Note]
> **Contributions are welcome. As highlighted in Appendix B.1 of our paper, several ambiguities can complicate binary categorization. While we have made every effort to ensure accuracy, we welcome community contributions to help maintain and refine this collection.**
>
> - Open an issue or PR to add new or missing papers.
> - Feel free to open an issue or [contact us](mailto:r12946015@ntu.edu.tw) if you think your paper would be better placed under a different categorization for learning rate, batch size, or rank tuning.
> - Thank you for helping us maintain a comprehensive and accurate survey!


## Survey Criteria

### Paper Inclusion Criteria

- The paper proposes a LoRA-based PEFT method.
- The primary objective of the proposed method is to enhance fine-tuning effectiveness, such as:
  - achieving higher accuracy under comparable trainable parameter counts; or
  - maintaining comparable performance with greater parameter efficiency.
- Vanilla LoRA is explicitly used as a baseline for performance comparison.
- The paper evaluates at least one decoder-only LLM.
- The paper satisfies at least one of the following conditions:
  - it was published in a major AI conference or journal;
  - it is a high-impact preprint with more than 40 citations; or
  - it was a preprint released recently, i.e., after 2025-06.


### Categorization Criteria for Hyperparameter Tuning

We determine whether key hyperparameters were tuned according to the following criteria:

- We rigorously verify whether the **vanilla LoRA baseline** underwent tuning. Therefore, tuning performed only for the proposed method or other advanced baselines is not counted as tuning.
- **Learning Rate (LR):** We mark LR tuning as positive only if vanilla LoRA is evaluated under at least three distinct learning rates.
- **Batch Size (BS):** We mark BS tuning as positive only if vanilla LoRA is evaluated under at least three distinct batch sizes.
- **Rank:** We mark Rank comparison as positive only if vanilla LoRA is evaluated under at least two distinct ranks.

Importantly:

- We assign a positive entry as long as the corresponding hyperparameter tuning or comparison is performed for at least one model-task combination in the paper.
- If the original authors explicitly state that a hyperparameter was tuned, we assign a positive entry. We do not verify whether the adopted search range covers the optimal configuration for vanilla LoRA, nor do we require explicit performance reports under all tested hyperparameter values, since such details are often unavailable.


## Comprehensive List of Papers

- ***arXiv Date*** marks the release date of the first version on arXiv (denoted as ***--*** if unavailable), with the corresponding link directing you to the paper's arXiv page. 
- ***Pub. Date*** refers to the formal publication date of the venue, where ***--*** indicates the paper has not yet been formally published. 
- The link on the publication date leads directly to the official conference or journal publication page. The table is sorted primarily by Pub. Date, followed by arXiv Date.


| Method | arXiv Date | Pub. Date | Venue | Decoder-only LLM | Fine-tuned Task | LR | BS | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **DyLoRA** | [2022-10](https://arxiv.org/abs/2210.07558) | [2023-05](https://aclanthology.org/2023.eacl-main.239/) | EACL | GPT-2 Medium | NLG | ❌ | ❌ | ✅ |
| **GLoRA** | [2023-06](https://arxiv.org/abs/2306.07967) | -- | arXiv | Llama-1-7B<br>Llama-2-7B | NLG | ❌ | ❌ | ❌ |
| **LoRA-FA** | [2023-08](https://arxiv.org/abs/2308.03303) | -- | arXiv | Llama-1-7B<br>Llama-2-7B | Commonsense | ❌ | ❌ | ❌ |
| **Laplace-LoRA** | [2023-08](https://arxiv.org/abs/2308.13111) | [2024-05](https://openreview.net/forum?id=FJiUyzOF1m) | ICLR | Llama-1-7B<br>Llama-2-7B | Commonsense | ❌ | ❌ | ❌ |
| **VeRA** | [2023-10](https://arxiv.org/abs/2310.11454) | [2024-05](https://openreview.net/forum?id=NjNfLdxr3A) | ICLR | GPT-2 Medium/Large<br>Llama-1-7B/13B<br>Llama-2-7B/13B | NLG<br>Instruction Following | ✅ | ❌ | ✅ |
| **BoFT** | [2023-11](https://arxiv.org/abs/2311.06243) | [2024-05](https://openreview.net/forum?id=7NzgkEdGyr) | ICLR | Llama-2-7B | Instruction Following<br>Math | ❌ | ❌ | ✅ |
| **MoRA** | [2024-05](https://arxiv.org/abs/2405.12130) | -- | arXiv | Llama-2-7B/13B | UUID<br>Math<br>Instruction Following | ✅ | ❌ | ✅ |
| **Delta-LoRA** | [2023-09](https://arxiv.org/abs/2309.02411) | -- | arXiv | GPT-2 Medium | NLG | ❌ | ❌ | ❌ |
| **Tied-LoRA** | [2023-11](https://arxiv.org/abs/2311.09578) | [2024-06](https://aclanthology.org/2024.naacl-long.481/) | NAACL | GPT-2B-001<br>Llama-2-7B | NLG<br>Commonsense<br>Math | ❌ | ❌ | ✅ |
| **LoRETTA** | [2024-02](https://arxiv.org/abs/2402.11417) | [2024-06](https://aclanthology.org/2024.naacl-long.174/) | NAACL | Llama-2-7B/13B/70B | NLG<br>GLUE | ❌ | ❌ | ❌ |
| **AutoLoRA** | [2024-03](https://arxiv.org/abs/2403.09113) | [2024-06](https://aclanthology.org/2024.naacl-long.282/) | NAACL | GPT-2 Medium | NLG | ❌ | ❌ | ❌ |
| **ALoRA** | [2024-05](https://arxiv.org/abs/2403.16187) | [2024-06](https://aclanthology.org/2024.naacl-long.35/) | NAACL | GPT-2 Large<br>Llama-2-7B | NLG<br>GLUE<br>Instruction Following | ❌ | ❌ | ✅ |
| **RoSA** | [2024-01](https://arxiv.org/abs/2401.04679) | [2024-07](https://proceedings.mlr.press/v235/nikdan24a.html) | ICML | Llama-2-7B | NLG<br>Math<br>Code<br>Instruction Following | ✅ | ❌ | ❌ |
| **LoRA+** | [2024-02](https://arxiv.org/abs/2402.12354) | [2024-07](https://proceedings.mlr.press/v235/hayou24a.html) | ICML | GPT-2<br>Llama-1-7B | GLUE<br>Instruction Following | ✅ | ❌ | ❌ |
| **scaled AdamW** | [2024-02](https://arxiv.org/abs/2402.02347) | [2024-07](https://proceedings.mlr.press/v235/zhang24ax.html) | ICML | GPT-2 Medium<br>Mistral-7B-v0.1 | NLG<br>GLUE | ✅ | ❌ | ✅ |
| **DoRA** | [2024-02](https://arxiv.org/abs/2402.09353) | [2024-07](https://proceedings.mlr.press/v235/liu24bn.html) | ICML | Llama-1-7B/13B<br>Llama-2-7B<br>Llama-3-8B | Commonsense | ❌ | ❌ | ✅ |
| **FLORA** | [2024-02](https://arxiv.org/abs/2402.03293) | [2024-07](https://proceedings.mlr.press/v235/hao24a.html) | ICML | GPT-2-base/XL | Summarization<br>Translation | ✅ | ❌ | ✅ |
| **FourierFT** | [2024-05](https://arxiv.org/abs/2405.03003) | [2024-07](https://proceedings.mlr.press/v235/gao24o.html) | ICML | GPT-2 Medium/Large<br>Llama-1-7B/13B<br>Llama-2-7B/13B | NLG<br>Instruction Following | ✅ | ✅ | ✅ |
| **ResLoRA** | [2024-02](https://arxiv.org/abs/2402.18039) | [2024-08](https://aclanthology.org/2024.findings-acl.525/) | ACL | Llama-2-7B | Math<br>Commonsense | ❌ | ❌ | ✅ |
| **PLoRA** | [2024-02](https://arxiv.org/abs/2402.16141) | -- | arXiv | Llama-1-7B | Instruction Following<br>Math | ✅ | ❌ | ✅ |
| **OLoRA** | [2024-06](https://arxiv.org/abs/2406.01775) | -- | arXiv | Mistral-7B<br>Llama-2-7B<br>Tiny Llama-1.1B<br>Gemma-2B<br>OPT-1.3B | Commonsense<br>Instruction Following | ❌ | ❌ | ✅ |
| **LamDA** | [2024-06](https://arxiv.org/abs/2406.12832) | [2024-11](https://aclanthology.org/2024.findings-emnlp.563/) | EMNLP | Llama-2-7B | NLG<br>Math<br>Commonsense | ❌ | ❌ | ✅ |
| **PiSSA** | [2024-04](https://arxiv.org/abs/2404.02948) | [2024-12](https://proceedings.neurips.cc/paper_files/paper/2024/hash/db36f4d603cc9e3a2a5e10b93e6428f2-Abstract-Conference.html) | NeurIPS | Llama-2-7B/13B<br>Llama-3-8B/70B<br>Mistral-7B-v0.1<br>Gemma-7B<br>Qwen1.5-7B<br>Yi-1.5-34B<br>DeepSeek-MoE-16B<br>Mixtral-8x7B | Math<br>Code<br>Instruction Following | ❌ | ❌ | ✅ |
| **VB-LoRA** | [2024-05](https://arxiv.org/abs/2405.15179) | [2024-12](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e0d38c676d5855bcfab7f6d29d20ad9-Abstract-Conference.html) | NeurIPS | GPT-2 Medium/Large<br>Llama-2-7B/13B<br>Mistral-7B-v0.1<br>Gemma-7B | NLG<br>Math<br>Instruction Following | ❌ | ❌ | ❌ |
| **HRA** | [2024-05](https://arxiv.org/abs/2405.17484) | [2024-12](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html) | NeurIPS | Llama-2-7B | Math | ❌ | ❌ | ❌ |
| **CorDA** | [2024-06](https://arxiv.org/abs/2406.05223) | [2024-12](https://proceedings.neurips.cc/paper_files/paper/2024/hash/83f95bb0ac5046338ea2afe3390e9f4b-Abstract-Conference.html) | NeurIPS | Llama-2-7B/13B<br>Llama-3-8B<br>Gemma-2-9B | Math<br>Code<br>Instruction Following<br>World Knowledge | ❌ | ❌ | ✅ |
| **LoRA-GA** | [2024-07](https://arxiv.org/abs/2407.05000) | [2024-12](https://proceedings.neurips.cc/paper_files/paper/2024/hash/62c4718cc334f6a0a62fb81c4a2095a1-Abstract-Conference.html) | NeurIPS | Llama-2-7B | Math<br>Code<br>Instruction Following | ❌ | ❌ | ❌ |
| **RoAd** | [2024-09](https://arxiv.org/abs/2409.00119) | [2024-12](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3dbcadb7beedc2afe32bb23f75dd30ec-Abstract-Conference.html) | NeurIPS | Llama-1-7B/13B<br>Llama-2-7B<br>Llama-3-8B | Math<br>Commonsense | ❌ | ❌ | ✅ |
| **LoRA-drop** | [2024-02](https://arxiv.org/abs/2402.07721) | [2025-01](https://aclanthology.org/2025.coling-main.371/) | COLING | Llama-2-7B | NLG<br>Summarization<br>GLUE<br>Math | ❌ | ❌ | ❌ |
| **AG-LoRA** | -- | [2025-01](https://ieeexplore.ieee.org/document/10852296) | IEEE Access | Llama-1-7B | Commonsense | ❌ | ❌ | ✅ |
| **LoRA-Pro** | [2024-07](https://arxiv.org/abs/2407.18242) | [2025-04](https://openreview.net/forum?id=gTwRMU3lJ5) | ICLR | Llama-1-7B<br>Llama-2-7B<br>Llama-3-8B<br>Llama-3.1-8B | Math<br>Code<br>Instruction Following | ✅ | ❌ | ✅ |
| **LoRA-Dash** | [2024-09](https://arxiv.org/abs/2409.01035) | [2025-04](https://openreview.net/forum?id=RYrJqz44p4) | ICLR | Llama-1-7B<br>Llama-2-7B<br>Llama-3-8B<br>Qwen2.5-7B | GLUE<br>Commonsense | ❌ | ❌ | ✅ |
| **KaSA** | [2024-09](https://arxiv.org/abs/2412.06071) | [2025-04](https://openreview.net/forum?id=OQqNieeivq) | ICLR | Llama-1-7B<br>Llama-2-7B<br>Llama-3-8B<br>Qwen2.5-7B | GLUE<br>Commonsense<br>Instruction Following | ❌ | ❌ | ✅ |
| **RandLoRA** | [2025-02](https://arxiv.org/abs/2502.00987) | [2025-04](https://openreview.net/forum?id=Hn5eoTunHN) | ICLR | GPT-2 Medium<br>Qwen2-0.5B<br>Phi3-3B<br>Llama-3-8B | NLG<br>Commonsense | ❌ | ❌ | ✅ |
| **DeLoRA** | [2025-03](https://arxiv.org/abs/2503.18225) | [2025-04](https://openreview.net/forum?id=X1U74IwuxG) | ICLR | Llama-2-7B<br>Llama-3-8B | Commonsense | ✅ | ❌ | ✅ |
| **HiRA** | -- | [2025-04](https://openreview.net/forum?id=TwJrTz9cRS) | ICLR | Llama-2-7B<br>Llama-3-8B | Math<br>Commonsense<br>Dialogue Generation | ❌ | ❌ | ✅ |
| **MiLoRA** | [2024-06](https://arxiv.org/abs/2406.09044) | [2025-04](https://aclanthology.org/2025.naacl-long.248/) | NAACL | Llama-2-7B<br>Llama-3-8B<br>Qwen2.5-7B | Math<br>Commonsense<br>Instruction Following | ❌ | ❌ | ❌ |
| **SSMLoRA** | [2025-02](https://arxiv.org/abs/2502.04958) | [2025-04](https://aclanthology.org/2025.naacl-long.230/) | NAACL | GPT-2<br>Llama-2-7B/13B | GLUE | ✅ | ❌ | ✅ |
| **LoRA-One** | [2025-02](https://arxiv.org/abs/2502.01235) | [2025-07](https://openreview.net/forum?id=KwIlvmLDLm) | ICML | Llama-2-7B | Math<br>Code<br>Instruction Following | ✅ | ✅ | ❌ |
| **Init[AB]** | [2025-05](https://arxiv.org/abs/2505.23194) | [2025-07](https://proceedings.mlr.press/v267/li25bm.html) | ICML | Llama-3-8B | Arithmetic<br>Commonsense | ✅ | ❌ | ❌ |
| **Lily** | [2024-07](https://arxiv.org/abs/2407.09946) | [2025-07](https://aclanthology.org/2025.findings-acl.874/) | ACL | Llama-3-8B | Commonsense | ❌ | ❌ | ❌ |
| **C3A** | [2024-07](https://arxiv.org/abs/2407.19342) | [2025-07](https://aclanthology.org/2025.findings-acl.102/) | ACL | Llama-2-7B<br>Llama-3-8B/70B<br>Mistral-7B<br>Mixtral-8x7B | Math<br>Code<br>Commonsense | ❌ | ❌ | ❌ |
| **SuLoRA** | -- | [2025-07](https://aclanthology.org/2025.findings-acl.278/) | ACL | Llama-2-7B | Instruction Following | ❌ | ❌ | ✅ |
| **BiDoRA** | [2024-10](https://arxiv.org/abs/2410.09758) | [2025-08](https://openreview.net/forum?id=v2xCm3VYl4) | TMLR | GPT-2 Medium | NLG | ❌ | ❌ | ❌ |
| **HD-PiSSA** | [2025-05](https://arxiv.org/abs/2505.18777) | [2025-11](https://aclanthology.org/2025.emnlp-main.330/) | EMNLP | Llama-2-7B<br>Llama-3-8B<br>Mistral-7B-v0.1 | Math<br>Code | ❌ | ❌ | ✅ |
| **LoSiA** | [2025-07](https://arxiv.org/abs/2507.04487) | [2025-11](https://aclanthology.org/2025.emnlp-main.340/) | EMNLP | Gemma-2B<br>Llama-2-7B/13B | Math<br>Code<br>Commonsense<br>Instruction Following | ✅ | ❌ | ✅ |
| **Sensitivity-LoRA** | [2025-09](https://arxiv.org/abs/2509.09119) | [2025-11](https://aclanthology.org/2025.findings-emnlp.709/) | EMNLP | GPT-2 Large<br>Qwen2.5-7B/32B<br>Llama-3.1-8B | NLG<br>Instruction Following | ❌ | ❌ | ❌ |
| **OHoRA** | -- | [2025-11](https://aclanthology.org/2025.emnlp-main.951/) | EMNLP | Llama-2-7B<br>Llama-3-8B<br>Gemma-7B<br>Llama-3.1-8B-Inst | Math<br>Code<br>Commonsense<br>Instruction Following | ❌ | ❌ | ✅ |
| **EVA** | [2024-10](https://arxiv.org/abs/2410.07170) | [2025-12](https://openreview.net/forum?id=movsqor65f) | NeurIPS | Llama-2-7B<br>Gemma-2-70B<br>Llama-3.1-8B/70B | Math<br>Code<br>Commonsense | ✅ | ❌ | ✅ |
| **GoRA** | [2025-02](https://arxiv.org/abs/2502.12171) | [2025-12](https://openreview.net/forum?id=d1dL1ymD6N) | NeurIPS | Llama-3.1-8B<br>Llama-2-7B | Math<br>Code<br>Instruction Following | ❌ | ❌ | ❌ |
| **AuroRA** | [2025-05](https://arxiv.org/abs/2505.18738) | [2025-12](https://openreview.net/forum?id=2hgHyoyVWj) | NeurIPS | Llama-3-8B | Commonsense | ❌ | ❌ | ✅ |
| **GraLoRA** | [2025-05](https://arxiv.org/abs/2505.20355) | [2025-12](https://openreview.net/forum?id=8wvOMQ2Olw) | NeurIPS | Llama-3.1-8B/70B<br>Llama-3.2-3B<br>Qwen-2.5-1B/7B | Math<br>Code<br>Commonsense | ❌ | ❌ | ✅ |
| **FlyLoRA** | [2025-10](https://arxiv.org/abs/2510.08396) | [2025-12](https://openreview.net/forum?id=nGQLYn13Xf) | NeurIPS | Llama-3.1-8B<br>Qwen-2.5-7B/14B | MMLU<br>Science<br>Math<br>Code | ❌ | ❌ | ✅ |
| **DropLoRA** | [2025-08](https://arxiv.org/abs/2508.17337) | -- | arXiv | Llama-2-7B<br>Llama-3-8B | Math<br>Code<br>Commonsense<br>Instruction Following | ❌ | ❌ | ✅ |
| **PrunedLoRA** | [2025-09](https://arxiv.org/abs/2510.00192) | -- | arXiv | Llama-3-8B | Math<br>Commonsense | ✅ | ❌ | ✅ |
| **LoRA-DA** | [2025-10](https://arxiv.org/abs/2510.24561) | -- | arXiv | Llama-2-7B | Math<br>Commonsense | ❌ | ❌ | ✅ |
| **ABM-LoRA** | [2025-11](https://arxiv.org/abs/2511.19145) | -- | arXiv | Llama-2-7B | Instruction Following | ❌ | ❌ | ✅ |
| **MiSS** | [2024-09](https://arxiv.org/abs/2409.15371) | [2026-04](https://openreview.net/forum?id=gohmWoUSoS) | ICLR | Llama-2-7B/13B<br>Mistral-7B<br>Qwen3-4B<br>Llama-3.2-3B | Math<br>Code<br>Instruction Following | ❌ | ❌ | ✅ |
| **LoFT** | [2025-05](https://arxiv.org/abs/2505.21289) | [2026-04](https://openreview.net/forum?id=86P3sb1dpr) | ICLR | GPT-2-base/Large<br>Llama-1-7B<br>Llama-2-7B<br>Llama-3-8B<br>Llama-3.1-70B | NLG<br>Math<br>Code<br>Commonsense | ❌ | ❌ | ✅ |
| **FlexLoRA** | [2026-01](https://arxiv.org/abs/2601.22905) | [2026-04](https://openreview.net/forum?id=tqnkbdYWWm) | ICLR | Llama-3-8B | Commonsense | ❌ | ❌ | ❌ |
| **Stable-LoRA** | [2026-03](https://arxiv.org/abs/2603.05204) | [2026-04](https://openreview.net/forum?id=xSa19DAieH) | ICLR | Qwen-2-0.5B/1B<br>Llama-1-7B<br>Llama-3.1-8B<br>Llama-3.2-1B/3B | Math<br>Commonsense | ✅ | ❌ | ❌ |
| **RaLoRA** | -- | [2026-04](https://openreview.net/forum?id=kObvnQ6pUx) | ICLR | Llama-3.1-8B | Math<br>Code<br>Instruction Following | ❌ | ❌ | ✅ |
| **GiVA** | [2026-04](https://arxiv.org/abs/2604.21901) | [2026-05](https://openreview.net/forum?id=80lrQEKKJg) | AISTATS | Qwen-2-0.5B<br>OLMo-2-7B<br>Phi-3-3.8B<br>Mistral-7B | Math<br>Code<br>Commonsense<br>Instruction Following | ✅ | ❌ | ❌ |
| **PEANuT** | [2024-10](https://arxiv.org/abs/2410.01870) | [2026-08](https://dl.acm.org/doi/abs/10.1145/3770854.3780230) | KDD | Llama-2-7B<br>Llama-3-8B<br>Qwen-3-8B | Math<br>Commonsense | ❌ | ❌ | ❌ |