import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
from peft import PeftModel
from trl import DPOTrainer, CPOTrainer
from trl import CPOConfig


def format_prompt(prompt, target_style):
    control_code = f"[{target_style.upper()}]" if target_style is not None else ""
    return f"{control_code}{prompt} [SEP] "

def get_p_reversed(scores, t_tss, t_sbert, t_cola):
    agg_scores = scores['tss_tgt'] ** t_tss * scores['sbert'] ** t_sbert * scores['cola'] ** t_cola
    min_cols = agg_scores.argsort(-1)[:, 0]
    max_cols = agg_scores.argsort(-1)[:, -1]
    row_indices = np.arange(max_cols.shape[0])
    p_reversed = {
        m: (scores[m][row_indices, min_cols] > scores[m][row_indices, max_cols]).mean()
        for m in ["tss_tgt", "sbert", "cola"]
    }
    return p_reversed

def get_rewards_temperatures(scores, t_max=10):
    t_tss, t_sbert, t_cola = 1, 1, 1
    p_reversed = get_p_reversed(scores, t_tss, t_sbert, t_cola)

    while (t_tss < t_max and 
           (p_reversed['tss_tgt'] > p_reversed['sbert'] or  
            p_reversed['tss_tgt'] > p_reversed['cola'])):
        t_tss += 1
        p_reversed = get_p_reversed(scores, t_tss, t_sbert, t_cola)
    
    while t_sbert <= t_max and p_reversed['sbert'] > p_reversed['tss_tgt']:
        t_sbert += 1
        p_reversed = get_p_reversed(scores, t_tss, t_sbert, t_cola)
    t_sbert = max(t_sbert - 1, 1)
    p_reversed = get_p_reversed(scores, t_tss, t_sbert, t_cola)
    
    while (t_cola <= t_max 
           and p_reversed['cola'] > p_reversed['sbert'] 
           and p_reversed['cola'] > p_reversed['tss_tgt']):
        t_cola += 1
        p_reversed = get_p_reversed(scores, t_tss, t_sbert, t_cola)
    t_cola = max(t_cola - 1, 1)

    return {"tss_tgt": t_tss, "sbert": t_sbert, "cola": t_cola}

def construct_po_pair(
    sample,
    tokenizer,
    reward_components=[],
    neg_sample_selection=None,
    model_score_temperature=None,
    fear_only_model_score=False,
    add_noise_to_fear=False,
    rewards_temperatures={},
):
    n_gen = len(sample['generated'])
    score_keys = list(sample['generated'][0].keys())
    score_keys.remove('txt')
    scores = {k: np.array([item[k] for item in sample['generated']]) for k in score_keys}
    scores = {k: np.where(v < 0, 0, v) for k, v in scores.items()}

    reward = np.ones(n_gen)
    fear_reward = -1 * np.ones(n_gen)
    for r in reward_components:
        if r == "mean_prob":
            weighted_model_score = np.power(scores[r], model_score_temperature)
            if not fear_only_model_score:
                reward += weighted_model_score
            fear_reward += weighted_model_score
        else:
            reward_t = rewards_temperatures.get(r, 1)
            reward *= scores[r] ** reward_t
            fear_reward *= scores[r] ** reward_t
    if add_noise_to_fear:
        gaussian_noise = np.random.normal(0, fear_reward.std(), n_gen)
        fear_reward += gaussian_noise
    
    sorted_ids = reward.argsort()
    pos_idx = sorted_ids[-1]
    fear_sorted_ids = fear_reward.argsort()

    if neg_sample_selection == "random":
        neg_idx = random.choice(sorted_ids[:-1])
    elif neg_sample_selection == "highest":
        neg_idx = sorted_ids[-2]
    elif neg_sample_selection == "lowest":
        neg_idx = sorted_ids[0]
    elif neg_sample_selection == "fear":
        neg_idx = fear_sorted_ids[-1]
    else:
        assert False
    po_sample = {
        "prompt": format_prompt(sample["prompt"], sample["tgt_style"]),
        "chosen": sample["generated"][pos_idx]["txt"],
        "rejected": sample["generated"][neg_idx]["txt"],
    }
    return po_sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--save_path', type=str, default="test")
    parser.add_argument('--hf_cache_dir', type=str, default=os.environ.get("TRANSFORMERS_CACHE", None))
    parser.add_argument('--base_model_name', type=str, default="meta-llama/Llama-2-7b-hf", help="llama model name or path")
    parser.add_argument('--adapter_dirs', nargs='+', type=str, required=True) 
    parser.add_argument('--po_algorithm', type=str, choices=["dpo", "cpo"], required=True)
    parser.add_argument('--reward_components', nargs='+')
    parser.add_argument('--model_score_temperature', type=float, default=1.0)
    parser.add_argument('--neg_sample_selection', type=str, choices=["random", "highest", "lowest", "fear"], required=True)
    parser.add_argument('--fear_only_model_score', action='store_true')
    parser.add_argument('--add_noise_to_fear', action='store_true')
    parser.add_argument('--weighted_rewards', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()

    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if "cds" in args.dataset:
        from utils.style_maps import CDS_STYLE_MAP as STYLE_MAP
    elif "ets" in args.dataset:
        from utils.style_maps import ETS_STYLE_MAP as STYLE_MAP
    elif "gyafc" in args.dataset:
        from utils.style_maps import GYAFC_STYLE_MAP as STYLE_MAP
    else:
        assert False

    if args.po_algorithm == "dpo":
        from trl import DPOTrainer as Trainer
        from trl import DPOConfig as TrainingArguments
    elif args.po_algorithm == "cpo":
        from trl import CPOTrainer as Trainer
        from trl import CPOConfig as TrainingArguments

    if args.weighted_rewards:
        assert (
            len(set(args.reward_components) & set(('tss_tgt', 'sbert', 'cola'))) == 3
        ), "Must use all 3 rewards for weighted aggregated reward."

    transformers.set_seed(args.random_seed)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    
    # load dataset
    dataset = defaultdict(list)
    styles = os.listdir(args.dataset)
    for style in styles:
        style_dataset = os.path.join(args.dataset, style)
        data_files = os.listdir(style_dataset)
        for data_file in data_files:
            split = data_file[:data_file.index('.')]
            with open(os.path.join(style_dataset, data_file)) as f:
                for line in f:
                    dataset[split].append(json.loads(line))
    scores = {
        r: np.array([[item[r] for item in line['generated']] for line in dataset['train']]) 
        for r in args.reward_components
    }
    if args.weighted_rewards:
        rewards_temperatures = get_rewards_temperatures(scores, t_max=6)
        p_reversed = get_p_reversed(
            scores, 
            rewards_temperatures['tss_tgt'], 
            rewards_temperatures['sbert'], 
            rewards_temperatures['cola'],
        )
    else:
        rewards_temperatures = {r: 1 for r in args.reward_components}
    if local_rank == 0:
        print(f"training data len={len(dataset['train'])}")
        print('\t'.join(['rewards_temps'] \
                    + [f"{k}={v:.2f}" for k, v in rewards_temperatures.items()]))
        if args.weighted_rewards:
            print('\t'.join(['p_reversed'] \
                + [f"{k}={v:.2f}" for k, v in p_reversed.items()])) 
    dataset = {k: [construct_po_pair(
        sample, 
        tokenizer,
        reward_components=args.reward_components, 
        neg_sample_selection=args.neg_sample_selection,
        fear_only_model_score=args.fear_only_model_score,
        add_noise_to_fear=args.add_noise_to_fear,
        model_score_temperature=args.model_score_temperature,
        rewards_temperatures=rewards_temperatures,
    ) for sample in v] for k, v in dataset.items()}
    dataset = {k: Dataset.from_pandas(pd.DataFrame(v)) for k, v in dataset.items()}

    # load base sft model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    # add adapters
    n_adapters = len(args.adapter_dirs)
    assert n_adapters >= 1
    if n_adapters == 1:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_dirs[0],
            adapter_name="combined",
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_dirs[0],
            adapter_name=str(0),
        )
        for i, adapter_dir in enumerate(args.adapter_dirs[1:]):
            model.load_adapter(adapter_dir, adapter_name=str(i+1))
        model.add_weighted_adapter(
            [str(i) for i in range(n_adapters)],
            [1.0] * n_adapters,
            combination_type="cat",
            adapter_name="combined",
        )
        model.set_adapter("combined")
        for i in range(n_adapters):
            model.delete_adapter(str(i))
    model = model.merge_and_unload()

    per_device_batch_size = int(args.batch_size / world_size)
    num_update_steps_per_epoch = int(len(dataset['train']) / (args.batch_size * args.gradient_accumulation_steps))
    save_steps = int(num_update_steps_per_epoch / 3)
    warmup_steps = num_update_steps_per_epoch if args.warmup_steps is None else args.warmup_steps
    if local_rank == 0:
        print(f"\nnum_update_steps_per_epoch: {num_update_steps_per_epoch}\nper_device_batch_size: {per_device_batch_size}\nsave_steps: {save_steps}\n")
    training_args = TrainingArguments(
        max_steps=0 if not args.debug else 3,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.n_epochs, 
        beta=args.beta,
        max_length=512,
        max_target_length=512,
        max_prompt_length=512,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        logging_steps=10,
        eval_steps=save_steps,
        save_steps=save_steps,
        output_dir=args.save_path,
        optim="adamw_torch",
        warmup_steps=warmup_steps,
        bf16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_num_workers=8,
        disable_tqdm=args.disable_tqdm,
        report_to="none",
    )

    trainer = Trainer(
        model,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=['q_proj', 'v_proj'],
            bias="none",
            task_type="CAUSAL_LM",
        ),
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=9)],
    )

    trainer.train()
    trainer.save_model(os.path.join(args.save_path, "checkpoint-best"))
