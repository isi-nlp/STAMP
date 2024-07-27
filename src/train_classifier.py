import os
import argparse
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import (
    set_seed,
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def encode_example(example, tokenizer):
    txt = example['txt']
    style_idx = STYLE_MAP[example['style']]

    encoded_input = tokenizer(
        txt,
        max_length=512, 
        truncation=True,
    )
    labels = torch.zeros((len(STYLE_MAP),), dtype=float)
    labels[style_idx] = 1
    
    return {
        "input_ids": encoded_input['input_ids'],
        "attention_mask": encoded_input["attention_mask"],
        "labels": labels,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/cds_sampled")
    parser.add_argument('--save_path', type=str, default="trained_models/cds/classifier")
    parser.add_argument('--hf_model_name', type=str, default="roberta-large")
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    set_seed(args.random_seed)
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if "cds" in args.dataset:
        from utils.style_maps import CDS_STYLE_MAP as STYLE_MAP
    elif "gyafc" in args.dataset:
        from utils.style_maps import GYAFC_STYLE_MAP as STYLE_MAP
    else:
        assert False

    # load model / tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.hf_model_name,
        problem_type='multi_label_classification',
        num_labels=len(STYLE_MAP),
    )

    # load data
    data_files = {
        "train": "train.jsonl",
        "valid": "valid.jsonl",
        "test": "test.jsonl",
    }

    dataset = load_dataset(
        args.dataset, 
        data_files=data_files, 
    )

    fn_kwargs = {"tokenizer": tokenizer}
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
            fp16=False,
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
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
        ),
    )

    trainer.train()
    trainer.save_model(os.path.join(args.save_path, "checkpoint-best"))