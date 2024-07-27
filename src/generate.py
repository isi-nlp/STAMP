import os
import json
import copy
import random
import argparse
from tqdm import tqdm

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from peft import PeftModel


def strip_output(txt):
    end_seqs = [f"[{style.upper()}]" for style in sorted(STYLE_MAP.keys())]
    end_seqs.append("[SEP]")
    for seq in end_seqs:
        if seq in txt:
            txt = txt[:txt.index(seq)]
    return txt.strip()

def generate(
    txts,
    model,
    tokenizer,
    generation_config,
    target_style=None,
    adapter_name=None,
    max_new_tokens=128,
):
    control_code = f"[{target_style.upper()}]" if target_style is not None else ""
    encoded_srcs = tokenizer(
        [f"{tokenizer.bos_token}{control_code}{txt}" for txt in txts],
        add_special_tokens=False,
        max_length=50,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    encoded_seps = tokenizer(
        ["[SEP]"] * len(txts), 
        add_special_tokens=False,
        padding=True,
        return_tensors='pt',
    )
    encoded_prompts = {k: torch.cat(
        (encoded_srcs[k], encoded_seps[k]), dim=1
    ).cuda() for k in encoded_srcs.keys()}
    input_len = encoded_prompts['input_ids'].size(1)
    if adapter_name is not None:
        model.set_adapter(adapter_name)
    output = model.generate(
        input_ids=encoded_prompts['input_ids'],
        attention_mask=encoded_prompts['attention_mask'],
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
    )
    output = tokenizer.batch_decode(output[:, input_len:], skip_special_tokens=True)
    output = [strip_output(txt) for txt in output]

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/cds/sampled/test.jsonl")
    parser.add_argument('--save_path', type=str, default="outputs/test.jsonl")
    parser.add_argument('--base_model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--adapter_dirs', nargs='+', type=str, default=[])
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--target_style', type=str, default=None)
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--greedy_decoding', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if "cds" in args.dataset:
        from utils.style_maps import CDS_STYLE_MAP as STYLE_MAP
    elif "ets" in args.dataset:
        from utils.style_maps import ETS_STYLE_MAP as STYLE_MAP
    elif "gyafc" in args.dataset:
        from utils.style_maps import GYAFC_STYLE_MAP as STYLE_MAP
    else:
        assert False

    transformers.set_seed(args.random_seed)

    # load model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.use_8bit,
        device_map=device_map
    )
    
    # load adapters
    n_adapters = len(args.adapter_dirs)
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
            adapter_name="0",
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

    # load data
    data = []
    with open(args.dataset) as f:
        for line in f:
            line = json.loads(line)
            if args.target_style is not None and line['style'] == args.target_style:
                continue
            data.append({
                'txt': line['txt'],
                'style': line['style'],
            })
    if args.debug:
        data = data[:32]
    output_data = copy.deepcopy(data)

    txts = [item['txt'] for item in data]

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
    )
    if args.greedy_decoding:
        generation_config = GenerationConfig(
            do_sample=False,
            repetition_penality=1.25,
        )

    save_dir = os.path.dirname(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    outputs = []
    for i in tqdm(range(0, len(txts), args.batch_size)):
        batch_txts = txts[i:i+args.batch_size]
        batch_outputs = generate(
            txts=batch_txts,
            model=model,
            tokenizer=tokenizer,
            target_style=args.target_style,
            adapter_name="combined",
            generation_config=generation_config,
            max_new_tokens=50,
        )
        outputs += batch_outputs

    with open(args.save_path, 'w') as f:
        for i, txt in enumerate(outputs):
            if args.target_style is not None:
                output_data[i]['transferred'] = txt
                output_data[i]['tgt_style'] = args.target_style
            else:
                output_data[i]['paraphrased'] = txt
            f.write(f"{json.dumps(output_data[i])}\n")
