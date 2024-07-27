import os
import json
import random
import argparse
from collections import defaultdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="data/gyafc/original")
    parser.add_argument('--output_dir', type=str, default="data/gyafc/sampled")
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)

    subsets = os.listdir(args.input_dir)

    data = defaultdict(lambda: defaultdict(list))
    for split in ['train', 'tune', 'test']:
        split_name = 'valid' if split == 'tune' else split
        for subset in subsets:
            file_paths = {k: os.path.join(args.input_dir, subset, split, k) for k in ['formal', 'informal']}
            for k, path in file_paths.items():
                with open(path) as f:
                    for line in f:
                        data[split_name][k].append({
                            'txt': line.strip(),
                            'style': k,
                        })

    sampled = {}
    for split in ['train', 'valid', 'test']:
        n = 2000 
        if split == 'valid':
            n = 200
        elif split == 'test':
            n = 1000
        data[split] = random.sample(data[split]['formal'], n) + random.sample(data[split]['informal'], n)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    for split in ["train", "valid", "test"]:
        with open(os.path.join(args.output_dir, f"{split}.jsonl"), 'w') as f:
            for line in data[split]:
                f.write(f"{json.dumps(line)}\n")