# Reference: https://github.com/tnurbek/loft/blob/main/hf_implementation.ipynb

from .loft_optim.optimizer import LoFTAdamW, AdamW
from .loft_optim.hf_callbacks import GradientClippingCallback
from transformers import TrainerCallback

def get_optimizer(model, lr, LoFT):
    # Use LoRAFTAdamW optimizer
    if LoFT:
        optimizer = LoFTAdamW(
            model.parameters(),
            lr=lr,
            # Below is not default AdamW values, so we turn them off
            # See https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
            # weight_decay=0.,
            # eps=1e-4,
            # betas=(0.9, 0.999),
            # But LoFTAdamW re-assign these values?
            model=model,
            lora_A_name='lora_A',
            lora_B_name='lora_B',
            alternate_update=True,
            rescale_grads=True,
            reproject_momentum=True,
            reproject_second_moment=True,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            # Below is not default AdamW values, so we turn them off
            # See https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
            # weight_decay=0.,
            # eps=1e-4,
            # betas=(0.9, 0.999),
        )
    return optimizer


class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])