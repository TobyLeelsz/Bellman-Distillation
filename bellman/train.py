import torch
import os
import json
import torch.distributed as dist
from accelerate import init_empty_weights


from arguments import get_args
from utils import print_args, initialize, get_tokenizer
from trainer import Trainer
from dataset import PromptDataset, LMPipeline


def main():
    
    args = get_args()
    initialize(args)

    device = torch.cuda.current_device()
    
    os.makedirs(args.save, exist_ok=True)
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
            
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    args.fp32 = not ds_config["fp16"]["enabled"]
    args.deepspeed_config = None
    
    if args.teacher_model_type is None:
        args.teacher_model_type = args.model_type
    
    tokenizer = get_tokenizer(args)

    eval_pipeline = PromptDataset(
        args, tokenizer, "valid", args.eval_path, num=args.dev_num
    )
    lm_pipeline = LMPipeline(
        args, tokenizer, "train", args.lm_data_dir, num=args.train_num
    )
    
    model = Trainer(args, tokenizer, ds_config)

    model.add_eval_pipeline(eval_pipeline)
    model.add_lm_pipeline(lm_pipeline)

    model.train_bellman()


if __name__ == "__main__":
    main()