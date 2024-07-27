dataset="data/cds/sft/"
save_root="trained_models/cds/transfer/sft"
base_model_name="meta-llama/Llama-2-7b-hf"
task="transfer"
learning_rate="5e-5"
n_epochs=12
batch_size=16
gradient_accumulation_steps=1
sbert_temperature=8.0

export TORCH_DISTRIBUTED_DEBUG=DETAIL

save_path="${save_root}"

cmd="torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:50500 \
    src/train_transfer_sft.py \
    --dataset $dataset \
    --save_path $save_path \
    --base_model_name $base_model_name \
    --n_epochs $n_epochs \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --sbert_temperature $sbert_temperature \
    --random_seed 42 \
    --disable_tqdm"

echo $cmd 
eval $cmd