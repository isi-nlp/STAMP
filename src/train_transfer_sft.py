import os
import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

import torch
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    EarlyStoppingCallback,
)
from transformers.file_utils import PaddingStrategy
from datasets import Dataset, DatasetDict
from datasets.utils.logging import disable_progress_bar
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
disable_progress_bar()


def print_trainable_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def select_best_sample(
    sample,
    sbert_temperature,
):
    n_gen = len(sample['generated'])
    score_keys = list(sample['generated'][0].keys())
    score_keys.remove('txt')
    scores = {k: np.array([item[k] for item in sample['generated']]) for k in score_keys}
    scores = {k: np.where(v < 0, 0, v) for k, v in scores.items()}

    scores['sbert'] = scores['sbert'] ** sbert_temperature
    reward = np.ones(n_gen)
    for r in ["tss_tgt", "sbert", "cola"]:
        reward *= scores[r]
    
    sorted_ids = reward.argsort()
    pos_idx = sorted_ids[-1]

    sample = {
        "txt": sample["prompt"],
        "transferred": sample["generated"][pos_idx]["txt"],
        "tgt_style": sample["tgt_style"]
    }
    return sample

def encode_example(example, tokenizer, max_len=50):
    src = example['txt']
    tgt = example['transferred']
    control_code = f"[{example['tgt_style'].upper()}]"

    encoded_src = tokenizer(
        f"{tokenizer.bos_token}{control_code}{src}",
        add_special_tokens=False,
        max_length=max_len,
        truncation=True,
    )
    encoded_sep = tokenizer("[SEP]", add_special_tokens=False)
    encoded_prompt = {k: encoded_src[k] + encoded_sep[k] for k in encoded_src.keys()}
    encoded_label = tokenizer(
        tgt,
        add_special_tokens=False,
        max_length=max_len,
        truncation=True,
    )
    encoded_label["input_ids"].append(tokenizer.eos_token_id)
    encoded_label["attention_mask"].append(1)

    labels = [-100] * len(encoded_prompt['input_ids']) + encoded_label['input_ids']
    
    return {
        "input_ids": encoded_prompt['input_ids'] + encoded_label['input_ids'],
        "attention_mask": encoded_prompt["attention_mask"] + encoded_label["attention_mask"],
        "labels": labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/paranmt_filtered")
    parser.add_argument('--save_path', type=str, default="trained_models/paranmt/paraphraser")
    parser.add_argument('--base_model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--use_gradient_checkpointing', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--sbert_temperature', type=float, default=10.0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    set_seed(args.random_seed)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

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
    dataset = {k: [select_best_sample(
        sample, 
        sbert_temperature=args.sbert_temperature,
    ) for sample in v] for k, v in dataset.items()}
    dataset = DatasetDict({k: Dataset.from_pandas(pd.DataFrame(v)) for k, v in dataset.items()})

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "right"

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.use_8bit,
        device_map=device_map,
    )

    # data preprocess
    fn_kwargs = {
        "tokenizer": tokenizer,
    }
    dataset = dataset.map(
        encode_example, 
        fn_kwargs=fn_kwargs, 
        num_proc=4,
        load_from_cache_file=False,
    )
    data_columns = [
        'input_ids', 
        'attention_mask',
        'labels', 
    ]
    dataset.set_format(
        columns=data_columns,
    )
    
    # load lora modules
    if args.use_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_gradient_checkpointing)
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj",
        "v_proj",
    ]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True,
    )

    per_device_batch_size = int(args.batch_size / world_size)
    num_update_steps_per_epoch = int(len(dataset['train']) / (args.batch_size * args.gradient_accumulation_steps))
    save_steps = int(num_update_steps_per_epoch / 3)
    warmup_steps = num_update_steps_per_epoch
    print(f"\nnum_update_steps_per_epoch: {num_update_steps_per_epoch}\nper_device_batch_size: {per_device_batch_size}\nsave_steps: {save_steps}\nwarmup_steps: {warmup_steps}\n")
    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        args=TrainingArguments(
            max_steps=0 if not args.debug else 3,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=args.n_epochs, 
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            output_dir=args.save_path,
            save_total_limit=2,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=True if ddp and not args.use_gradient_checkpointing else False,
            group_by_length=False,
            disable_tqdm=args.disable_tqdm,
            dataloader_num_workers=8,
            remove_unused_columns=False,
        ),
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=9)],
    )

    trainer.train()
    trainer.save_model(os.path.join(args.save_path, "checkpoint-best"))