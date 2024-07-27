cmd="python src/data_process/sample_and_process_gyafc.py \
    --input_dir data/gyafc/original \
    --output_dir data/gyafc/sampled \
    --random_seed 42"

echo $cmd
eval $cmd
