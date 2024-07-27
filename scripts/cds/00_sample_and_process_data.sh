cmd="python src/data_process/sample_and_process_cds.py \
    --input_dir data/cds/original \
    --output_dir data/cds/sampled \
    --random_seed 42"

echo $cmd
eval $cmd
