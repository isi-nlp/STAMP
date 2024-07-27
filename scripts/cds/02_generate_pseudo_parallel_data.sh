export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH=src

temperature=0.5
cmd="python src/generate_pseudo_parallel_data.py \
    --dataset data/cds/sampled \
    --splits train valid \
    --save_path data/cds/pseudo_parallel \
    --base_model_name meta-llama/Llama-2-7b-hf \
    --adapter_dirs trained_models/paraphraser/checkpoint-best \
    --temperature ${temperature} \
    --top_p 1.0 \
    --batch_size 4 \
    --n_responses_per_query 20"
echo $cmd
eval $cmd