dataset="data/gyafc/pseudo_parallel"
save_root="trained_models/gyafc/strap"
base_model_name="meta-llama/Llama-2-7b-hf"
task="transfer"
learning_rate="5e-5"
n_epochs=6
batch_size=8
gradient_accumulation_steps=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL

for tgt_style in "formal" "informal"
do
    save_path="${save_root}/${tgt_style}"
    cmd="torchrun \
        --nnodes=1 \
        --nproc_per_node=2 \
        --max_restarts=0 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:50002 \
        src/sft.py \
        --dataset $dataset \
        --save_path $save_path \
        --base_model_name $base_model_name \
        --task $task \
        --target_style $tgt_style \
        --n_epochs $n_epochs \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --learning_rate $learning_rate \
        --random_seed 42 \
        --disable_tqdm"

    echo $cmd 
    eval $cmd
done