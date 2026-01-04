from typing import Dict
import numpy as np
import os
import time
import torch.distributed as dist
from torch.distributed import get_rank
import random
import torch
import torch.nn as nn
from datetime import timedelta
import deepspeed
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    )


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)

def initialize(args):
    # init bmt
    if args.deepspeed:
        init_distributed_ds(args)
    else:
        init_distributed(args)

    set_random_seed(args.seed, args.model_parallel)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)

def get_rev_kl_prime(logp, logq):
    q = torch.exp(logq)
    kl = q * (logq - logp)
    return kl.sum(-1)

def get_tokenizer(args, sample=False):
    if sample:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif args.model_type=="qwen":
        tokenizer.pad_token_id = 151646
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer

def save_rank(log_str, save_path, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        with open(save_path, "a") as f:
            f.write(log_str + "\n")


def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)

def init_distributed_ds(args):
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    deepspeed.init_distributed(timeout=timedelta(minutes=300))

def set_random_seed(seed, mp=False):
    """Set random seed for reproducability."""
    seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if mp:
            mpu.model_parallel_cuda_manual_seed(seed)

def get_optimizer_params(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln_f.weight', 'ln_1.weight', 'ln_2.weight', 'ln_cross_attn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters


def get_optimizer_params_peft(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if p.requires_grad]},
    ]

    return optimizer_grouped_parameters

def get_model(args, device):
    config = AutoConfig.from_pretrained(args.model_path)
    # pdb.set_trace()
    st_time = time.time()
    config.is_model_parallel = False
    if args.model_type=="qwen":
        dtype = torch.float32 if args.fp32 else torch.float16
    else:
        dtype = torch.float32 if args.fp32 else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, device_map={"": device}, torch_dtype=dtype)

    if args.peft is not None:
        if args.peft == "lora":

            model.enable_input_require_grads()
            if args.peft_path is not None:
                model = PeftModel.from_pretrained(model, args.peft_path)
            else:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=(not args.do_train), r=args.peft_lora_r, lora_alpha=args.peft_lora_alpha, lora_dropout=args.peft_lora_dropout
                )
                model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            raise NotImplementedError
    else:
        if dist.get_rank() == 0:
            print(' > number of parameters: {}'.format(
                sum([p.nelement() for p in model.parameters()])), flush=True)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    ed_time = time.time()
    
    print_rank(f"Model load time: {ed_time - st_time}s")
    
    return model

def all_gather(t, dim=0, world_size=None, group=None, op="cat"):
    if world_size is None:
        world_size = dist.get_world_size()
    all_t = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(all_t, t, group=group)
    if op == "cat":
        all_t = torch.cat(all_t, dim=dim)
    elif op == "stack":
        all_t = torch.stack(all_t, dim=dim)
    elif op == "list":
        all_t = all_t
    return all_t

def get_rev_kl(log_p, log_q, mask):  
    log_ratio = (log_p - log_q) * mask
    kl = log_ratio.float().exp() - 1 - log_ratio 
    return kl