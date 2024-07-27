export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=src

temperature=0.7
for style in "aae" "bible" "coha_1810" "coha_1890" "coha_1990" "joyce" "lyrics" "poetry" "shakespeare" "switchboard" "tweets"
do
    cmd="python src/generate_transfer_data.py \
        --dataset data/cds/pseudo_parallel \
        --splits train valid \
        --save_path data/cds/sft/${style} \
        --input_key paraphrased \
        --base_model_name meta-llama/Llama-2-7b-hf \
        --adapter_dirs trained_models/cds/strap/${style}/checkpoint-best \
        --cls_model_dir trained_models/cds/classifier/checkpoint-best \
        --target_style ${style} \
        --temperature ${temperature} \
        --top_p 1.0 \
        --batch_size 1 \
        --n_responses_per_query 90"

    echo $cmd
    eval $cmd
done