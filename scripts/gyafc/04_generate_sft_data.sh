export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=src

temperature=0.7
for style in "formal" "informal"
do
    cmd="python src/generate_transfer_data.py \
        --dataset data/gyafc/pseudo_parallel \
        --splits train valid \
        --save_path data/gyafc/sft/${style} \
        --input_key paraphrased \
        --base_model_name meta-llama/Llama-2-7b-hf \
        --adapter_dirs trained_models/gyafc/strap/${style}/checkpoint-best \
        --cls_model_dir trained_models/gyafc/classifier/checkpoint-best \
        --target_style ${style} \
        --temperature ${temperature} \
        --top_p 1.0 \
        --batch_size 1 \
        --n_responses_per_query 90"

    echo $cmd
    eval $cmd
done