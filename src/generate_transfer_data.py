import os
import json
import copy
import random
import argparse
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
    parser.add_argument('--dataset', type=str, default="data/reddit/transfer")
    parser.add_argument('--splits', nargs='+')
    parser.add_argument('--save_path', type=str, default="test/test.jsonl")
    parser.add_argument('--base_model_name', type=str, default="meta-llama/Llama-2-7b-hf", help="llama model name or path")
    parser.add_argument('--cls_model_dir', type=str, required=True)
    parser.add_argument('--cola_model_name', type=str, default="cointegrated/roberta-large-cola-krishna2020")
    parser.add_argument('--sbert_model_name', type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument('--input_key', type=str, required=True, choices=["txt", "paraphrased"])
    parser.add_argument('--adapter_dirs', nargs='+', type=str, default=[])
    parser.add_argument('--target_style', type=str, required=True)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_input_len', type=int, default=512)
    parser.add_argument('--n_responses_per_query', type=int, default=2)
    parser.add_argument('--use_8bit', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
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

    # load style reward model (style classifier)
    cls_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    cls_model = AutoModelForSequenceClassification.from_pretrained(
        args.cls_model_dir
    ).eval().to(device)

    # load fluency reward model (cola)
    cola_tokenizer = AutoTokenizer.from_pretrained(args.cola_model_name)
    cola = AutoModelForSequenceClassification.from_pretrained(args.cola_model_name).to(device)

    # load data
    data = defaultdict(lambda: defaultdict(list))
    data_files = os.listdir(args.dataset)
    for data_file in data_files:
        if data_file.split('.')[-1] not in ['txt', 'json', 'jsonl']:
            continue
        split = data_file[:data_file.index('.')]
        with open(os.path.join(args.dataset, data_file)) as f:
            for line in f:
                line = json.loads(line)
                if line['style'] != args.target_style:
                    data[split][line['style']].append(line)
        styles = sorted(data[split].keys())
        for style in styles:
            if 'gyafc' in args.dataset:
                data[split][style] = random.sample(data[split][style], 2000 if split=="train" else 200)
            else:
                data[split][style] = random.sample(data[split][style], 200 if split=="train" else 20)
        data[split] = [item for style in styles for item in data[split][style]]
    assert len(data["train"]) == 2000
    assert len(data["valid"]) == 200

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

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for split in args.splits:
        outputs = defaultdict(list)
        with open(os.path.join(args.save_path, f"{split}.jsonl"), 'w') as f:
            for i in tqdm(range(0, len(data[split]), args.batch_size), disable=args.disable_tqdm):
                batch = data[split][i:i+args.batch_size]
                batch = {k: [item[k] for item in batch] for k in batch[0].keys()}

                # set tgt_style
                batch['tgt_style'] = [args.target_style] * len(batch['style'])
                assert all([s != t for s, t in zip(batch['style'], batch['tgt_style'])])
                
                assert batch['tgt_style'][0] == args.target_style
                assert len(set(batch['tgt_style'])) == 1
                # generate
                batch_response, batch_repsonse_score = generate(
                    txts=batch[args.input_key],
                    model=model,
                    adapter_name="combined",
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    max_new_tokens=64,
                )
                batch_repsonse_score = batch_repsonse_score.cpu().tolist()

                # compute style scores
                batch_encoded_cls = cls_tokenizer(batch_response, max_length=50, truncation=True, padding=True, return_tensors='pt')
                batch_encoded_cls = {k: v.to(device) for k, v in batch_encoded_cls.items()}
                with torch.no_grad():
                    batch_cls_outputs = cls_model(**batch_encoded_cls).logits
                batch_cls_preds = torch.sigmoid(batch_cls_outputs).cpu().detach()

                src_tgt_ids = torch.tensor(
                    [[STYLE_MAP[src_style], STYLE_MAP[tgt_style]]
                    for src_style, tgt_style in zip(batch['style'], batch['tgt_style'])]
                ).repeat_interleave(args.n_responses_per_query, 0).cpu()
                src_tgt_scores = torch.gather(batch_cls_preds, 1, src_tgt_ids).cpu().tolist()

                # compute sbert cossims
                src_sbert_embeds = sbert.encode(
                    batch['txt'],
                    convert_to_tensor=True,
                ).repeat_interleave(args.n_responses_per_query, 0)
                res_sbert_emebds = sbert.encode(batch_response, convert_to_tensor=True)
                sbert_cossims = F.cosine_similarity(src_sbert_embeds, res_sbert_emebds).tolist()

                # compute cola scores
                batch_encoded_cola = cola_tokenizer(batch_response, max_length=50, truncation=True, padding=True, return_tensors='pt')
                batch_encoded_cola = {k: v.to(device) for k, v in batch_encoded_cola.items()}
                with torch.no_grad():
                    batch_cola_outputs = cola(**batch_encoded_cola).logits
                cola_scores = batch_cola_outputs.softmax(1)[:, 0].cpu().detach().tolist()

                # compute length penalty
                src_lens = torch.tensor([len(tokenizer.tokenize(src)) for src in batch['txt']], dtype=float)
                src_lens = src_lens.repeat_interleave(args.n_responses_per_query, dim=0)
                res_lens = torch.tensor([len(tokenizer.tokenize(res)) for res in batch_response], dtype=float)
                lp = torch.exp(1 - torch.min(src_lens, res_lens) / torch.max(src_lens, res_lens)).tolist()

                # write data
                for i in range(len(batch['txt'])):
                    line = {
                        "prompt": batch['txt'][i],
                        "paraphrased": batch['paraphrased'][i],
                        "style": batch['style'][i],
                        "tgt_style": batch['tgt_style'][i],
                        "generated": [
                            {
                                "txt": batch_response[j],
                                "tss_src": src_tgt_scores[j][0],
                                "tss_tgt": src_tgt_scores[j][1],
                                "sbert": sbert_cossims[j],
                                "cola": cola_scores[j],
                                "lp": lp[j],
                                "mean_prob": batch_repsonse_score[j],
                            } for j in range(i*args.n_responses_per_query, (i+1)*args.n_responses_per_query)
                        ], 
                    }
                    f.write(f"{json.dumps(line)}\n")
                    