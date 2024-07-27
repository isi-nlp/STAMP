export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=src

n_gpus=2
device_list_str=$(seq 0 $((n_gpus - 1)) | paste -sd, -)

start_time=$(date +%s)
styles=("formal" "informal")
base_model_name="meta-llama/Llama-2-7b-hf"
po_sample_strategy="fear"
po_algo="cpo"
n_epochs="16"
po_algo_param="default"
po_adapter_root="trained_models/gyafc/transfer/cpo/${po_algo_param}"
sft_adapter_path="trained_models/gyafc/transfer/sft/checkpoint-best"
dataset_dir="data/gyafc/sampled"
output_root="outputs/gyafc/transfer/cpo"
validation_output_root="outputs/gyafc/transfer/cpo_validation"

for iter_i in {0..10}
do
    echo -e "\n\n------------------------------------- ITER ${iter_i} -------------------------------------\n"
    curr_time=$(date +%s)
    time_elapsed=$((curr_time - start_time))
    echo -e "ITER ${iter_i} START TIME: ${time_elapsed} s\n\n"
    po_data_root="data/gyafc/cpo/${po_algo_param}/round_${iter_i}/${po_sample_strategy}"
    po_adapter_path="${po_adapter_root}/round_${iter_i}/${po_sample_strategy}"

    for style_i in "${!styles[@]}"
    do
        style="${styles[$style_i]}"
        adapter_dirs=${sft_adapter_path}
        for j in $(seq 0 $((iter_i - 1))); do
            adapter_dirs+=" ${po_adapter_root}/round_${j}/${po_sample_strategy}/checkpoint-best"
        done
        cmd="python src/generate_po_data.py \
        --dataset ${dataset_dir} \
        --splits train valid \
        --save_path ${po_data_root}/${style} \
        --base_model_name $base_model_name \
        --adapter_dirs ${adapter_dirs} \
        --cls_model_dir trained_models/gyafc/classifier/checkpoint-best \
        --target_style ${style} \
        --temperature 1.0 \
        --top_p 1.0 \
        --batch_size 8 \
        --n_responses_per_query 10 \
        --random_seed 42 \
        --disable_tqdm"

        gpu_id=$((style_i % n_gpus))
        cmd="CUDA_VISIBLE_DEVICES=$gpu_id ${cmd} &"
        echo $cmd
        eval $cmd
        
        if [ $((style_i % n_gpus)) -eq $((n_gpus-1)) ]; then
            wait
        fi
    done
    wait

    adapter_dirs=${sft_adapter_path}
    for j in $(seq 0 $((iter_i - 1))); do
        adapter_dirs+=" ${po_adapter_root}/round_${j}/${po_sample_strategy}/checkpoint-best"
    done
    cmd="CUDA_VISIBLE_DEVICES=${device_list_str} torchrun \
        --nnodes=1 \
        --nproc_per_node=${n_gpus} \
        --max_restarts=0 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:50041 \
        src/train_po.py \
        --po_algorithm ${po_algo} \
        --reward_components tss_tgt sbert cola \
        --neg_sample_selection fear \
        --weighted_rewards \
        --dataset ${po_data_root} \
        --base_model_name ${base_model_name} \
        --adapter_dirs $adapter_dirs \
        --save_path ${po_adapter_path} \
        --n_epochs ${n_epochs} \
        --batch_size 16 \
        --gradient_accumulation_steps 1 \
        --beta 0.1 \
        --lr 2e-6 \
        --random_seed 42 \
        --disable_tqdm"

    echo $cmd 
    eval $cmd

    dataset="${dataset_dir}/test.jsonl"
    output_dir="${output_root}/${po_algo_param}/round_${iter_i}/${po_sample_strategy}"
    for style_i in "${!styles[@]}"
    do
        style="${styles[$style_i]}"
        save_path="${output_dir}/${style}.jsonl"
        adapter_dirs=${sft_adapter_path}
        for j in $(seq 0 $iter_i); do
            adapter_dirs+=" ${po_adapter_root}/round_${j}/${po_sample_strategy}/checkpoint-best"
        done
        cmd="python src/generate.py \
            --dataset $dataset \
            --save_path $save_path \
            --base_model_name $base_model_name \
            --target_style $style \
            --adapter_dirs $adapter_dirs \
            --batch_size 32 \
            --temperature 0.7 \
            --top_p 1.0 \
            --random_seed 42"

        gpu_id=$((style_i % n_gpus))
        cmd="CUDA_VISIBLE_DEVICES=$gpu_id ${cmd} &"
        echo $cmd
        eval $cmd
        
        if [ $((style_i % n_gpus)) -eq $((n_gpus-1)) ]; then
            wait
        fi
    done
    wait

    dataset="${dataset_dir}/valid.jsonl"
    output_dir="${validation_output_root}/${po_algo_param}/round_${iter_i}/${po_sample_strategy}"
    for style_i in "${!styles[@]}"
    do
        style="${styles[$style_i]}"
        save_path="${output_dir}/${style}.jsonl"
        adapter_dirs=${sft_adapter_path}
        for j in $(seq 0 $iter_i); do
            adapter_dirs+=" ${po_adapter_root}/round_${j}/${po_sample_strategy}/checkpoint-best"
        done
        cmd="python src/generate.py \
            --dataset $dataset \
            --save_path $save_path \
            --base_model_name $base_model_name \
            --target_style $style \
            --adapter_dirs $adapter_dirs \
            --batch_size 32 \
            --temperature 0.7 \
            --top_p 1.0 \
            --random_seed 42"

        gpu_id=$((style_i % n_gpus))
        cmd="CUDA_VISIBLE_DEVICES=$gpu_id ${cmd} &"
        echo $cmd
        eval $cmd
        
        if [ $((style_i % n_gpus)) -eq $((n_gpus-1)) ]; then
            wait
        fi
    done
    wait

    curr_time=$(date +%s)
    time_elapsed=$((curr_time - start_time))
    echo -e "\nITER ${iter_i} END TIME: ${time_elapsed} s\n\n"
done