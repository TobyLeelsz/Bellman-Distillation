import os
import struct
import shutil
import random
import pdb
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from itertools import accumulate
from torch.distributed import get_rank, get_world_size

import numpy as np
import torch
import torch.distributed as dist
from utils import print_rank, save_rank
from tqdm import tqdm
import json



dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16,
    9: np.uint32
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


class DistributedMMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'
        def __init__(self, path):
            with open(path, 'rb') as stream:
                
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                # pdb.set_trace()
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset)
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, name, rank_number, rank_total, cache = None):
        
        super().__init__()

        self._path = path
        self._name = name
        self._state = 0
        if cache is not None:
            self._cache = cache
            os.makedirs(self._cache, exist_ok=True)
        else:
            self._cache = None
        self._rank_total = rank_total
        self._rank_number = rank_number
        self._index = None
        self._bin_buffer = None
        self._bin_buffer_mmap = None
        self.max_state, self.history = self._probe_data_path(self._path, self._name, self._rank_total)
        self.total_length = self.history[self.max_state-1][1]

        self._do_init(self._path, self._name, self._cache, self._state)

    def _probe_data_path(self, path, name, rank_total):
        print_rank("Probing Dataset")
            
        state = 0
        history = {-1:(0, 0)}
        for state in range(np.iinfo(np.int32).max):
            source_file = path + name + f"_{state}"
            if self.exists(source_file):
                index = self.Index(index_file_path(source_file))
                history[state] = (history[state-1][1], history[state-1][1] + len(index))
            else:
                break
            
        print_rank(f"Probing end. Max data state {state}, total length {history[state-1][1]}")
        
        return state, history

    def __getstate__(self):
        return self._path + self._name + "_%d"%(self._state)

    def __setstate__(self, state):
        self._state = state
        self._do_init(self._path, self._name, self._cache, self._state)

    def _do_init(self, path, name, cache, state):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        if self._index is not None:
            del self._index

        self._state = state

        source_file = path + name + f"_{self._state}"
        self._index = self.Index(index_file_path(source_file))
        self._bin_buffer_mmap = np.memmap(data_file_path(source_file), mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        if self._index is not None:
            del self._index

    def __len__(self):
        return self.total_length

    def _next_file(self):
        self._state += 1
        if self._state >= self.max_state:
            self._state = 0
        # print_rank(f"next_file: {self._state}")
        self._do_init(self._path, self._name, self._cache, self._state)
    
    def __relative_idx(self, idx):
        res = idx - self.history[self._state][0]
        return res

    def __slice_item(self, start, stop):
        ptr = self._index._pointers[self.__relative_idx(start)]
        sizes = self._index._sizes[self.__relative_idx(start):self.__relative_idx(stop)]
        offsets = list(accumulate(sizes))
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=sum(sizes), offset=ptr)
        return np.split(np_array, offsets[:-1])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            while idx >= self.history[self._state][1] or idx < self.history[self._state][0]:
                self._next_file()
            ptr, size = self._index[self.__relative_idx(idx)]
            return np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
        elif isinstance(idx, slice):
            raise NotImplementedError()

    @property
    def sizes(self):
        return self._index.sizes
        
    def exists(self, path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )


class BellmanOfflineDataset():
    def __init__(self, args, tokenizer, split="train", ppo_data_path=None, fix_prompts=False, num=-1):
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.rng_ppo = random.Random(args.seed_ppo)
        self.min_prompt_length = args.min_prompt_length
        self.max_prompt_length = args.max_prompt_length

        self.ppo_ctx = DistributedMMapIndexedDataset(ppo_data_path, f"{split}", get_rank(), get_world_size())
        self.ppo_raw, self.ppo_answers = None, None
        if os.path.exists(os.path.join(ppo_data_path, f"{split}.jsonl")):
            with open(os.path.join(ppo_data_path, f"{split}.jsonl")) as f:
                self.ppo_raw = [json.loads(line) for line in f.readlines()]
                self.ppo_answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.ppo_raw]

        self.num = min(num, len(self.ppo_ctx)) if num > 0 else len(self.ppo_ctx)
        self.fix_prompts = fix_prompts
        self.prompt_lengths = [None for _ in range(num)]
        print_rank(f"Dataset size: {len(self.ppo_ctx)}")
        print_rank(f"Duplicate count: {self.count_duplicates()}")
        # pdb.set_trace()

    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        data = self.ppo_ctx[index].astype(int)
        return data

    def create_loader(self, batch_size: int, shuffle=False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        
        sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last, rank=dp_rank, num_replicas=dp_world_size)
        return DataLoader(
            self, sampler=sampler, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers
        )

    def collate(self, samples):
        bs = len(samples)
        
        max_prompt_length = self.max_prompt_length
        
        model_batch = {
            "query_tensors": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "teacher_responses": torch.ones(bs, self.max_length - self.max_prompt_length, dtype=torch.long) * self.pad_id,
        } # modified for full query response
        
        for i, data in enumerate(samples):
            # left padding
            model_batch["query_tensors"][i] = torch.tensor(data[:self.max_prompt_length], dtype=torch.long)
            model_batch["teacher_responses"][i] = torch.tensor(data[self.max_prompt_length + 1:], dtype=torch.long)
        
        return model_batch
    
    def move_to_device(self, model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)          
        
        return model_batch

    def move_from_device(self, model_batch):
        for k in model_batch:
            model_batch[k] = model_batch[k].cpu()
        return model_batch

    def count_duplicates(self):
        """
        Calculate the number of duplicated (query_tensor, teacher_response) pairs in the dataset.
        A pair is considered duplicated if both query_tensor and teacher_response match exactly.
        """
        seen_pairs = set()
        duplicate_count = 0

        for i in range(len(self)):
            data = self.__getitem__(i)
            query_tensor = tuple(data[:self.max_prompt_length])
            teacher_response = tuple(data[self.max_prompt_length + 1:])
            pair = (query_tensor, teacher_response)

            if pair in seen_pairs:
                duplicate_count += 1
            else:
                seen_pairs.add(pair)

        return duplicate_count

class BellmanPipeline():
    def __init__(self, args, tokenizer, split, ppo_data_path=None, fix_prompts=False, num=-1):
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.rng_ppo = random.Random(args.seed_ppo)
        self.min_prompt_length = args.min_prompt_length
        self.max_prompt_length = args.max_prompt_length

        self.ppo_ctx = DistributedMMapIndexedDataset(ppo_data_path, f"{split}", get_rank(), get_world_size())
        self.ppo_raw, self.ppo_answers = None, None
        if os.path.exists(os.path.join(ppo_data_path, f"{split}.jsonl")):
            with open(os.path.join(ppo_data_path, f"{split}.jsonl")) as f:
                self.ppo_raw = [json.loads(line) for line in f.readlines()]
                self.ppo_answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.ppo_raw]

        self.num = min(num, len(self.ppo_ctx)) if num > 0 else len(self.ppo_ctx)
        self.fix_prompts = fix_prompts
        self.prompt_lengths = [None for _ in range(num)]
        print_rank(f"Num PPO instances: {len(self.ppo_ctx)}")
            
    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        data = self.ppo_ctx[index].astype(int)
        
        # assert len(data) <= self.max_prompt_length
        
        if self.args.model_type!="qwen" and 65535 in data:
            source_len = np.where(data==65535)[0][0]
            prompt = data[:source_len]
            response = data[source_len+1:]
        else:
            prompt = data
            response = None
        
        # return prompt, rest
        return prompt, response
    
    def collate(self, samples):
        bs = len(samples)
        
        max_prompt_length = self.max_prompt_length
        
        model_batch = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        } # modified for full query response
        
        no_model_batch = {
            "full_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "full_attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
            "full_label_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * -100,
        }
        # pdb.set_trace()
        for i, (prompt, response) in enumerate(samples):
            # left padding
            model_batch["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            model_batch["attention_mask"][i][-len(prompt):] = 1
            if response is not None:
                full_ids = np.concatenate([prompt, response], axis=0)
                no_model_batch["full_ids"][i][:len(full_ids)-1] = torch.tensor(full_ids[:-1], dtype=torch.long)
                no_model_batch["full_attention_mask"][i][:len(full_ids)-1] = 1.0
                no_model_batch["full_label_ids"][i][len(prompt)-1:len(full_ids)-1] = torch.tensor(response, dtype=torch.long)
        
        return model_batch, no_model_batch

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)        
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch

    def create_loader(self, batch_size: int, shuffle=False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        
        sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last, rank=dp_rank, num_replicas=dp_world_size)
        return DataLoader(
            self, sampler=sampler, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers
        )


class PromptDataset(Dataset):
    def __init__(self, args, tokenizer, split, data_path=None, num=-1):
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.min_prompt_length = args.min_prompt_length
        self.max_prompt_length = args.max_prompt_length

        if args.bin_data:
            self.data = DistributedMMapIndexedDataset(data_path, f"{split}", get_rank(), get_world_size())
        elif args.json_data:
            self.data, self.origin_data = self.load_data_json(data_path)
        else:
            # txt data
            self.data = self.load_data_txt(data_path)
        
        if os.path.exists(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")):
            with open(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]
        elif os.path.exists(os.path.join(data_path, f"{split}.jsonl")):
            with open(os.path.join(data_path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]
        else:
            print_rank("WARNING: No answers exist")
            
        self.label_map = {tokenizer.encode(x[0], add_special_tokens=False)[0]: x[0] for x in self.answers}
            
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print_rank(f"Num instances: {len(self.data)}")
            
    def __len__(self):
        return self.num

    def load_data_json(self, data_path):
        if os.path.exists(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")):
            data_path = os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")
        else:
            data_path = os.path.join(data_path, f"{self.split}.jsonl")
        
        with open(data_path) as f:
            lines = f.readlines()
        data_origin = [json.loads(line) for line in lines]
        data = []
        print_rank("Loading Data")
        for d in tqdm(data_origin, disable=(get_rank() != 0)):
            prompt = d["prompt"].replace("<n>", "\n")
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            output_ids = None
            if "output" in d:
                if isinstance(d["output"], list):
                    output_ids = self.tokenizer.encode(d["output"][0])
                else:
                    output_ids = self.tokenizer.encode(d["output"])
            data.append({
                "prompt_ids": prompt_ids,
                "output_ids": output_ids[:self.max_length - self.max_prompt_length]
            })
        print_rank("Load End")
        return data, data_origin

    def load_data_txt(self, data_path):
        with open(os.path.join(data_path, f"{self.split}.txt")) as f:
            lines = f.readlines()
        data = []
        print_rank("Loading Data")
        for line in lines:
            line = line.strip()
            line = line.replace("<n>", "\n")
            prompt = self.tokenizer.encode(line)
            data.append(prompt)
        print_rank("Load End")
        return data

    def verbalizer(self):
        return self.label_map

    def __getitem__(self, index: int):
        data = self.data[index]
        if self.args.bin_data:
            data = data.astype(int)
        elif self.args.json_data:
            output_ids = data["output_ids"]
            data = data["prompt_ids"]
        
        prompt_length = self.max_prompt_length

        prompt = data[:prompt_length]
        rest = data[prompt_length:]  
        if self.args.json_data:
            if output_ids is not None:
                rest = output_ids  
    
        return index, prompt, rest
    
    def collate(self, samples):
        bs = len(samples)
        
        max_prompt_length = self.max_prompt_length
        max_rest_length = max([len(samp[2]) for samp in samples])
        
        model_batch = {
            "input_ids": torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_prompt_length, dtype=torch.long),
            # "position_ids": torch.zeros(bs, max_prompt_length, dtype=torch.long)
        }
        
        no_model_batch = {
            "idx": torch.zeros(bs, dtype=torch.long),
            "rest_ids": torch.ones(bs, max_rest_length, dtype=torch.long) * self.pad_id
        }
        
        for i, (idx, prompt, rest) in enumerate(samples):
            # left padding
            model_batch["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            model_batch["attention_mask"][i][-len(prompt):] = 1
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            no_model_batch["idx"][i] = idx
            no_model_batch["rest_ids"][i][:len(rest)] = torch.tensor(rest, dtype=torch.long)
        
        return model_batch, no_model_batch

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)        
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch

    def create_loader(self, batch_size: int, shuffle=False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        
        sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last, rank=dp_rank, num_replicas=dp_world_size)
        return DataLoader(
            self, sampler=sampler, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers
        )


class LMPipeline():
    def __init__(self, args, tokenizer, split, lm_data_path=None, num=-1):
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.rng_lm = random.Random(args.seed_lm)

        self.lm_ctx = DistributedMMapIndexedDataset(lm_data_path, f"{split}", get_rank(), get_world_size())
        self.num = min(num, len(self.lm_ctx)) if num > 0 else len(self.lm_ctx)
        print_rank(f"Num LM instances: {len(self.lm_ctx)}")
            
    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self._get_lm(index)

    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids[:self.max_length]
        }

    def _process_lm(self, i, samp, model_data, no_model_data):
        input_ids = samp["input_ids"]
        source_len = 1
        
        if self.args.model_type!="qwen" and 65535 in input_ids:
            source_len = np.where(input_ids==65535)[0][0]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        input_ids = input_ids[:self.max_length]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["label"][i][:source_len-1] = -100
        no_model_data["loss_mask"][i][:input_len-1] = 1.0
        no_model_data["loss_mask"][i][:source_len-1] = 0

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)

        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch

    def collate(self, samples):
        bs = len(samples)
        
        max_length = self.max_length
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length, dtype=torch.long)
        }

        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(bs, max_length, dtype=torch.long)

        no_model_data = {
            "label": torch.ones(bs, self.max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }
        
        for i, samp in enumerate(samples):        
            self._process_lm(i, samp, model_data, no_model_data)
            
        return model_data, no_model_data

    def create_loader(self, batch_size: int, shuffle=False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        
        sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last, rank=dp_rank, num_replicas=dp_world_size)
        return DataLoader(
            self, sampler=sampler, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers
        )
