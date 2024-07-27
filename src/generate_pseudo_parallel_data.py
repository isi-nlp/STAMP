import os
import json
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GenerationConfig,
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer


def strip_output(txt):
    if '[SEP]' in txt:
        txt = txt[:txt.index('[SEP]')]
    return txt.strip()

def generate(
    txts,
    model,
    tokenizer,
    generation_config,
    adapter_name=None,
    max_new_tokens=64,
):
    encoded_srcs = tokenizer(
        [f"{tokenizer.bos_token}{txt}" for txt in txts],
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
    ).to(model.device) for k in encoded_srcs.keys()}
    input_len = encoded_prompts['input_ids'].size(1)
    if adapter_name is not None:
        model.set_adapter(adapter_name)
    output = model.generate(
        input_ids=encoded_prompts['input_ids'],
        attention_mask=encoded_prompts['attention_mask'],
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
    )
    scores = torch.cat(output.scores, dim=0).reshape(len(output.scores), output.scores[0].size(0), -1).transpose(0, 1)
    seqs = output.sequences[:, input_len:]
    seq_scores = torch.gather(F.softmax(scores, -1), 2, seqs.unsqueeze(-1)).squeeze().clone()
    seq_scores[seqs == 0] = 0
    seq_scores = seq_scores.mean(-1)
    decoded_seqs = tokenizer.batch_decode(seqs, skip_special_tokens=True)
    decoded_seqs = [strip_output(txt) for txt in decoded_seqs]

    return decoded_seqs, seq_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="data/test")
    parser.add_argument('--splits', nargs='+')
    parser.add_argument('--save_path', type=str, default="test/test.jsonl")
    parser.add_argument('--base_model_name', type=str, default="meta-llama/Llama-2-7b-hf", help="llama model name or path")
    parser.add_argument('--sbert_model_name', type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument('--adapter_dirs', nargs='+', type=str, default=[])
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_input_len', type=int, default=512)
    parser.add_argument('--n_responses_per_query', type=int, default=2)
    parser.add_argument('--greedy_decoding', action='store_true')
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.greedy_decoding:
        assert args.n_responses_per_query == 1

    transformers.set_seed(args.random_seed)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
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
            adapter_name=f"combined",
        )
        model.set_adapter("combined")
        for i in range(n_adapters):
            model.delete_adapter(str(i))
        
    device = model.device

    # load semantic reward model (sbert)
    sbert = SentenceTransformer(args.sbert_model_name).to(device)
    sbert.max_seq_length = 512

    # load data
    data = defaultdict(list)
    data_files = os.listdir(args.dataset)
    for data_file in data_files:
        if data_file.split('.')[-1] not in ['txt', 'json', 'jsonl']:
            continue
        split = data_file[:data_file.index('.')]
        with open(os.path.join(args.dataset, data_file)) as f:
            for line in f:
                line = json.loads(line)
                data[split].append(line)
    if 'gyafc' in args.dataset:
        assert len(data["train"]) == 4000
        assert len(data["valid"]) == 400
    else:
        assert len(data["train"]) == 22000
        assert len(data["valid"]) == 2200

    if args.debug:
        for split in data.keys():
            data[split] = data[split][:8]

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
        num_return_sequences=args.n_responses_per_query,
    )
    if args.greedy_decoding:
        generation_config = GenerationConfig(
            do_sample=False,
            repetition_penality=1.25,
        )

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for split in args.splits:
        outputs = defaultdict(list)
        with open(os.path.join(args.save_path, f"{split}.jsonl"), 'w') as f:
            for i in tqdm(range(0, len(data[split]), args.batch_size), disable=args.disable_tqdm):
                batch = data[split][i:i+args.batch_size]
                batch_output = copy.deepcopy(batch)
                batch = {k: [item[k] for item in batch] for k in batch[0].keys()}

                # generate
                batch_response, batch_repsonse_score = generate(
                    txts=batch['txt'],
                    model=model,
                    adapter_name="combined",
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    max_new_tokens=64,
                )
                batch_response[1] = batch['txt'][1]

                # compute sbert cossims
                src_sbert_embeds = sbert.encode(
                    batch['txt'],
                    convert_to_tensor=True,
                ).repeat_interleave(args.n_responses_per_query, 0)
                res_sbert_emebds = sbert.encode(batch_response, convert_to_tensor=True)
                sbert_cossims = F.cosine_similarity(
                    src_sbert_embeds, 
                    res_sbert_emebds
                ).reshape(len(batch['txt']), args.n_responses_per_query)
                
                # write data
                input_arr = np.array(batch['txt'])
                output_arr = np.array(batch_response).reshape(len(batch['txt']), args.n_responses_per_query)
                io_same = input_arr[:, np.newaxis] == output_arr
                sbert_cossims[io_same] = 0
                max_values, selected_ids = sbert_cossims.max(-1)
                max_values = max_values.tolist()
                selected_ids = selected_ids.cpu().numpy()
                selected_paraphrases = output_arr[np.arange(selected_ids.shape[0]), selected_ids].tolist()

                for j, line in enumerate(batch_output):
                    line['paraphrased'] = selected_paraphrases[j]
                    line['sbert_cossim'] = max_values[j]
                    f.write(f"{json.dumps(line)}\n")
                    