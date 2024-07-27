import os
import json
import random
import argparse
from collections import defaultdict

import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="data/cds")
    parser.add_argument('--output_dir', type=str, default="data/cds_sampled")
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')

    subset_names = sorted(os.listdir(args.input_dir))
    data = defaultdict(list)
    for subset_name in subset_names:
        folder = os.path.join(args.input_dir, subset_name)
        for split in ["train", "dev", "test"]:
            lines = []
            full_path = os.path.join(folder, f"{split}.input0.bpe")
            with open(full_path) as f:
                txts = f.read().strip().split("\n")
            n_samples = 2000 if split == "train" else 200
            sampled_txts = random.sample(txts, n_samples)
            sampled_txts = [roberta.bpe.decode(x) for x in sampled_txts]
            for txt in sampled_txts:
                if subset_name == "aae":
                    txt = txt.replace('\\n', '\n').replace('\\', '')
                txt = txt.strip()
                data[split].append({"txt": txt, "style": subset_name})
    data["valid"] = data.pop("dev")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    for split in ["train", "valid", "test"]:
        with open(os.path.join(args.output_dir, f"{split}.jsonl"), 'w') as f:
            for line in data[split]:
                f.write(f"{json.dumps(line)}\n")
