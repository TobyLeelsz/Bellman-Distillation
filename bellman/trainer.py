import torch
from dataset import BellmanOfflineDataset, BellmanPipeline, LMPipeline
from losses import Loss
from typing import Optional, Tuple
import deepspeed
import os
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,)

import torch.distributed as dist
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from utils import *
import torch.nn.functional as F
from model import BellmanModel
import wandb
from rouge_metric import compute_metrics
import time


class Trainer():
    def __init__(self, args, tokenizer, ds_config):
        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.cuda.current_device()
        self.teacher_model = self.get_teacher_model()
        self.model = BellmanModel(self.args, self.device)
        self.ds_config = ds_config
        self.opt = self.get_optimizer()
        self.path = self.args.prompt_data_dir
        self.losses = Loss(self.args, self)
        self.scheduler = self.setup_scheduler()
        self.dataset = self.load_dataset()
        self.top_p = args.top_p
        self.max_length = args.max_length
        # pdb.set_trace()
        print("DeepSpeed Config:",self.ds_config)
        self.model, self.opt, self.scheduler = self.setup_ds(self.model, self.opt, self.scheduler)
        self.score = 0
        self.dp_world_size = dist.get_world_size()
        self.dp_rank = dist.get_rank()
        self.dp_group = None
        
        if dist.get_rank() == 0:
            print(' > number of parameters: {}M'.format(
                int(sum([p.nelement() for p in self.model.parameters()]) / 1e6)), flush=True)
        self.ds_config = ds_config
        self.state_value_head = None

        self.generate_kwargs = dict(
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            max_length=args.max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def forward_model(self, batch):
        outputs = self.model(
            **batch,
            return_dict=True,
            use_cache=False,
        )
        return outputs

    def broadcast_mp(self, batch, src=0, group=None):
        for k, v in batch.items():
            dist.broadcast(batch[k], src=src, group=group)
        
    def setup_scheduler(self):
        """
        Returns a learning rate scheduler derived from an instance's TRLConfig
        """
        if self.args.scheduler_name == "constant_trm":
            scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=self.args.warmup_iters)
        elif self.args.scheduler_name == "cosine_trm":
            scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=self.args.warmup_iters, num_training_steps=self.args.total_iters)
        else:
            scheduler_class = get_scheduler_class(self.args.scheduler_name)
            scheduler = scheduler_class(self.opt, eta_min=self.args.lr_min, T_max=self.args.total_iters)
        
        return scheduler

    def get_model_inputs(
        self,
        query_tensors,
        response_tensors,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.cat((query_tensors, response_tensors), dim=1)[
            :, -self.max_length :
        ]
        attention_mask = self.get_mask(tokens)
  
        batch = {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }
        
        if self.args.model_type in ["gpt2"]:  
            # For a proper positional encoding in case of left padding
            position_ids = attention_mask.cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask.eq(0), 0)
            batch["position_ids"] = position_ids
        
        return batch

    def get_mask(self, tokens):
        attention_mask = (
            tokens.not_equal(self.tokenizer.pad_token_id).long()
        )
        return attention_mask
    
    def compute_logits_and_log_probs(self, query_ids, response_ids, inf_mask=None, base="base", return_logprobs=True, top_p=None, return_value=False):
        batch = self.get_model_inputs(
            query_ids, response_ids
        )
        
        if base == "base":
            model_cls = self.model.module.forward
        elif base == "teacher":
            model_cls = self.teacher_model
        else:
            raise NotImplementedError

        outputs = model_cls(
            **batch,
            return_dict=True,
            use_cache=False
        )

        logits = outputs.logits
        logits = logits / self.args.temperature

        start = query_ids.size(1) - 1
        end = query_ids.size(1) + response_ids.size(1) - 1
        logits = logits[:, start:end]

        if inf_mask is not None:
            logits = logits.masked_fill(inf_mask, -float("inf"))
        
        if top_p is not None:
            filter_value = -1e3
            min_tokens_to_keep = 1
            sorted_logits, sorted_indices = torch.sort(logits, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -min_tokens_to_keep :] = 0
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
            top_p_logits = logits.masked_fill(indices_to_remove, filter_value)

            top_p_mask = (top_p_logits != filter_value)

            return top_p_logits, top_p_mask, logits

    def load_dataset(self):
        return BellmanOfflineDataset(self.args, self.tokenizer, split="train", ppo_data_path=self.path)

    def add_eval_pipeline(self, eval_pipeline: BellmanPipeline):
        """Adds pipeline from with validation prompts"""
        self.eval_pipeline = eval_pipeline

    def add_lm_pipeline(self, lm_pipeline: LMPipeline):
        """Adds pipeline from with validation prompts"""
        self.lm_pipeline = lm_pipeline

    def get_teacher_model(self):
        config = AutoConfig.from_pretrained(self.args.teacher_model_path)
        config.is_model_parallel = False
        model = AutoModelForCausalLM.from_pretrained(
            self.args.teacher_model_path, 
            config=config, 
            device_map={"": self.device}, 
            torch_dtype=torch.float16 if self.args.model_type!="qwen" else torch.bfloat16
        )

        if self.args.peft is not None:
            if self.args.peft == "lora":
                assert self.args.teacher_peft_path is not None
                model = PeftModel.from_pretrained(model, self.args.peft_path)
            else:
                raise NotImplementedError
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}'.format(
                    sum([p.nelement() for p in model.parameters()])), flush=True)

        model.eval()

        return model

    def setup_ds(self, model, optimizer=None, scheduler=None):
        if self.args.model_type=="qwen" and self.ds_config['fp16']['enabled']==True:
            import copy
            self.ds_config['bf16']=copy.deepcopy(self.ds_config['fp16'])
            self.ds_config['fp16']['enabled']=False
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=self.args,
            lr_scheduler=scheduler,
            mpu=None,
            config_params=self.ds_config
        )
        return model, optimizer, scheduler

    def get_optimizer(self):

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=[0.9, 0.95],
            eps=1.0e-8,
            weight_decay=1.0e-6
        )

        return optimizer

    def create_loader(self):
        raise NotImplementedError

    def train_bellman(self):
        self.prepare_learning()
        self.iter_count = 1
        self.global_iter_count = 1
        self.nth_evaluation = 0
        os.environ['WANDB_MODE'] = 'online'
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        local_rank = dist.get_rank()
        if get_rank() == 0:
            wandb.login()
            wandb.init(project="bellman-distill-340M", name="top_p_0.8", mode="online")
            start_time = time.time()


        print_rank("Total Steps:", self.total_steps, "Data Epochs:", self.args.epochs)
        lm_epochs = 0        
        self.save()
        
        for training_epoch in range(3):
            freq = {}
            # logits_softmax = []
            for it, batch in enumerate(self.train_dataloader):
                # pdb.set_trace()   
                if self.lm_pipeline is not None:
                    try:
                        lm_batch = next(self.lm_iterator)
                    except StopIteration:
                        lm_epochs += 1
                        print_rank(f"Another lm epoch, lm epochs: {lm_epochs}")
                        save_rank(f"Another lm epoch, lm epochs: {lm_epochs}", os.path.join(self.args.save, "log.txt"))
                        self.lm_dataloader.sampler.set_epoch(lm_epochs)
                        self.lm_iterator = iter(self.lm_dataloader)
                        lm_batch = next(self.lm_iterator)

                batch["query_tensors"] = batch["query_tensors"].to(self.device)
                batch["teacher_responses"] = batch["teacher_responses"].to(self.device)

                self.lm_pipeline.move_to_device(*lm_batch, self.device)
                stats = {}
                self.teacher_model.eval()
                
                if self.args.gradient_checkpointing:
                    self.model.module.base_model.gradient_checkpointing_enable()
                
                teacher_top_p_logits, teacher_top_p_mask, teacher_logits = self.compute_logits_and_log_probs(batch["query_tensors"], batch["teacher_responses"], base="teacher", top_p=self.top_p)
                student_top_p_logits, student_top_p_mask, student_logits = self.compute_logits_and_log_probs(batch["query_tensors"], batch["teacher_responses"], base="base", top_p=self.top_p)

                reg_phi_loss = self.losses.kl_loss(batch, student_logits, teacher_logits)

                bellman_loss, stats = self.losses.bellman_distill_loss(batch, student_logits, teacher_logits, teacher_top_p_mask) # NOTE: modified teacher to student

                lm_logits = self.forward_model(lm_batch[0]).logits

                pt_loss, pt_loss_stats = self.losses.pt_loss(lm_batch, lm_logits)

                stats.update(pt_loss_stats)

                with torch.no_grad():
                    stats["reg_kl_loss"] = reg_phi_loss.item()
                    mask = stats["mask"]
                loss = bellman_loss + self.args.lm_coef * pt_loss

                if get_rank() == 0:
                    elapsed = time.time() - start_time
                    stats["time"] = elapsed
                    wandb.log(stats)
           
                self.model.backward(loss)
                self.model.step()

                if self.args.gradient_checkpointing:
                    self.model.module.base_model.gradient_checkpointing_disable()

                if self.iter_count % self.args.gradient_accumulation_steps == 0 and \
                    ((self.global_iter_count < 10000 and (self.global_iter_count % 1000 == 0)) or \
                    self.global_iter_count % self.args.save_interval == 0):
                    self.save()

                # eval
                if self.iter_count % self.args.gradient_accumulation_steps == 0 and \
                    ((self.global_iter_count < 1000 and (self.global_iter_count % 100 == 0)) or \
                    (self.global_iter_count % self.args.eval_interval == 0)):
                    self.evaluate()
                
                self.iter_count += 1
                if self.iter_count % self.args.gradient_accumulation_steps == 0:
                    self.global_iter_count += 1
                    self.model.zero_grad()

        wandb.finish()


    def prepare_learning(self):
        self.train_dataloader = self.dataset.create_loader(
            self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True
        )

        self.eval_dataloader = self.eval_pipeline.create_loader(
            self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=False)

        self.lm_dataloader = self.lm_pipeline.create_loader(
            self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
        self.lm_iterator = iter(self.lm_dataloader)

        self.n_updates_per_batch = self.args.ppo_epochs
        self.total_steps = int(
            self.args.training_epochs
            * self.n_updates_per_batch
            * len(self.train_dataloader)
            / self.args.gradient_accumulation_steps
        )
        self.total_steps = min(self.total_steps, self.args.total_iters)

    def save(self, directory: Optional[str] = None):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        ckpt_dir = os.path.join(base_ckpt_path, f"{self.global_iter_count}-{self.score}")
        os.makedirs(ckpt_dir, exist_ok=True)
        print("save called!")
        if get_rank() == 0:
            self.model.module.base_model.save_pretrained(ckpt_dir, safe_serialization=False)
            print(f"Model save to {ckpt_dir}")
            self.tokenizer.save_pretrained(ckpt_dir)

    def generate(self, input_ids, attention_mask=None, mode="base", teacher_mixed_sample=False, top_p=1.0, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.device)[:, :self.args.max_prompt_length]
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)[:, :self.args.max_prompt_length]

        kwargs = dict(self.generate_kwargs, **kwargs)

        if mode == "base":
            model = self.model.module
        elif mode == "teacher":
            model = self.teacher_model
        else:
            raise NotImplementedError

        mix_in_model, mix_in_alpha = None, None
        if teacher_mixed_sample:
            mix_in_model = self.teacher_model
            mix_in_alpha = self.args.teacher_mixed_alpha

        with torch.no_grad():
            
            generation_config = GenerationConfig(**kwargs)

            # pdb.set_trace()
            
            max_new_tokens = generation_config.max_length - input_ids.size(1)

            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                mix_in_model=mix_in_model,
                mix_in_alpha=mix_in_alpha,
                top_p=top_p            
            )
            
            gen.sequences = F.pad(
                gen.sequences,
                (0, self.max_length - gen.sequences.shape[1]),
                value=self.tokenizer.pad_token_id,
            )
            
            if gen.scores is not None:
                gen.scores = torch.stack(gen.scores, dim=1)
                gen.scores = torch.cat([
                    gen.scores, 
                    torch.zeros(
                        gen.scores.size(0),
                        self.max_length - self.args.max_prompt_length - gen.scores.size(1),
                        gen.scores.size(2),
                        device=gen.scores.device)],
                    dim=1)

        return gen

    def evaluate(self):
        eval_results = {}
        eval_rl_results, preds, response_texts = self.evaluate_bellman()
        eval_results.update(eval_rl_results)
        
        if get_rank() == 0:
            res = compute_metrics(response_texts, self.eval_pipeline.answers)
            eval_results.update(res)
            wandb.log(eval_results)
            self.score = res["rougeL"]
        self.save()

    def evaluate_bellman(self): 
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        stats = {}
        all_full_ids = []
        all_rev_kl = []
        all_lens = []
        
        table = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, "Generation Evaluation", disable=(not get_rank() == 0)):
                batch, no_model_batch = batch
                batch, _ = self.eval_pipeline.move_to_device(batch, no_model_batch, self.device)
                gen_out = self.generate(
                    **batch,
                    return_dict_in_generate=True,
                    output_scores=True,
                    top_p=self.top_p
                )
                full_ids = gen_out.sequences
                gen_logits = gen_out.scores
                inf_mask = torch.isinf(gen_logits)

                all_full_ids.append(full_ids)
                
                input_ids = batch["input_ids"]
                gen_ids = full_ids[:, input_ids.size(1):]
                mask = self.get_mask(full_ids)
                mask = mask[:, input_ids.size(1)-1:input_ids.size(1)+gen_ids.size(1)-1]
                lens = torch.sum(mask, dim=-1)
                all_lens.append(lens)

            all_full_ids = torch.cat(all_full_ids, dim=0)
            # all_rev_kl = torch.cat(all_rev_kl, dim=0)
            all_lens = torch.cat(all_lens, dim=0)

            full_ids = all_gather(all_full_ids, dim=1, world_size=self.dp_world_size, group=self.dp_group, op="stack")
            full_ids = full_ids.view(-1, full_ids.size(-1))

            prompt_ids = full_ids[:, :self.eval_pipeline.max_prompt_length]
            all_lens = all_gather(all_lens, dim=0, world_size=self.dp_world_size, group=self.dp_group)
            stats["lens"] = all_lens.float().mean()

            response_texts = []
            if get_rank() == 0:
                prompt_texts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                response_texts = self.tokenizer.batch_decode(full_ids[:, self.eval_pipeline.max_prompt_length:], skip_special_tokens=True)
                gen_texts = [p + g for p, g in zip(prompt_texts, response_texts)]

                columns = ["prompts"]
                columns_data = [prompt_texts]
                columns.append("samples")
                if isinstance(gen_texts[0], str):
                    columns_data.append(gen_texts)
                else:
                    columns_data.append(gen_texts.tolist())

                table.append(list(zip(*columns_data)))

        # Log and display evaluation metrics
        if get_rank() == 0:
            rows = sum(list(map(list, zip(*table))), [])

        self.nth_evaluation += 1
        return stats, table, response_texts
