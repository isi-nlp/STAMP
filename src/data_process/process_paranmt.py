import os
import json
import pickle
import random
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="data/paranmt_filtered")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir

    random.seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for split in ["train", "dev"]:
        full_path = os.path.join(args.input_dir, f"{split}.pickle")
        with open(full_path, 'rb') as f:
            raw_data = pickle.load(f)
        processed = []
        for line in raw_data:
            txt = line[3]
            paraphrased = line[4]
            if random.random() <= 0.5:
                txt, paraphrased = paraphrased, txt
            processed.append({
                "txt": txt,
                "paraphrased": paraphrased,
            })
        
        split = "valid" if split == "dev" else split
        with open(os.path.join(args.output_dir, f"{split}.jsonl"), 'w') as f:
            for line in processed:
                f.write(f"{json.dumps(line)}\n")